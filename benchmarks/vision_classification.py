import os
import random

import numpy as np
import torch

# import matplotlib.pyplot as plt

from utils.function_utils import Augmentation
from utils.function_utils import UtilsAugmentations
from utils.function_utils import UtilsDataset

import utils.tta_policies
# from utils.function_utils import (
#     Augmentation,
#     UtilsAugmentations,
#     UtilsDataset,
# )


class VisionClassificationBenchmark:
    """
    Shared image-classification benchmark workflow for the conformance prediction framework.
    """

    # benchmark_name = None
    # model_file_prefix = None

    def __init__(
        self,
        # augmentations,
        path_logs,
        policy_name,
        list_other_aug=None, 
        seed=42,
        robust_training=False,
        train_fraction=1.0,
        test_fraction=0.4,
        calibration_fraction=0.5,
        precalibration_fraction=0.1,
        batch_size=256,
        iterations=5,
        # num_workers=0,
        probability_batch_size=256,
        heldout_precal=True,
        pretrained=True,
    ):
        
        # self.augmentations = augmentations
        # self.list_augmentations = list(augmentations.keys())
        self.calibration_data = None
        self.testing_data = None
        self.precalibration_data = None
        self.accuracies = {}
        self.class_accuracies = {}
        self.seed = seed
        self.path_logs = path_logs
        self.robust_training = robust_training
        self.batch_size = batch_size
        self.iterations = iterations
        # self.num_workers = num_workers
        self.probability_batch_size = probability_batch_size
        self.pretrained = pretrained
        

        # self.benchmark_name = None
        # self.model_file_prefix = None
        # self.data_root = None
        resolved_augmentations = {}

        if policy_name is not None:
            resolved_augmentations.update(
                utils.tta_policies.build_augs_dict(policy=policy_name)
            )

        # Backward compatibility with the existing interface.
        # if augmentations is not None:
        #     resolved_augmentations.update(augmentations)

        if list_other_aug is not None:
            resolved_augmentations.update(list_other_aug)

        self.augmentations = resolved_augmentations
        self.list_augmentations = list(resolved_augmentations.keys())
        self.list_augmentation_names = self.list_augmentations

        if not self.list_augmentations:
            raise ValueError("At least one augmentation must be provided.")

        if self.list_augmentations[0] != Augmentation.ORIGINAL.value:
            raise ValueError("Augmentations must start with ORIGINAL.")

        if not self.list_augmentations:
            raise ValueError('At least one augmentation must be provided.')
        
        fractions = {
            "train_fraction": train_fraction,
            "test_fraction": test_fraction,
            "calibration_fraction": calibration_fraction,
            "precalibration_fraction": precalibration_fraction,
        }

        for name, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")

        if heldout_precal:
            heldout_fraction = (test_fraction + calibration_fraction + precalibration_fraction)
            if heldout_fraction > 1.0 + 1e-9:
                raise ValueError("With heldout_precal=True, precalibration_fraction + calibration_fraction + test_fraction must be <= 1.")
        else:
            if test_fraction + calibration_fraction > 1.0 + 1e-9:
                raise ValueError("calibration_fraction + test_fraction must be <= 1.")

            if precalibration_fraction > train_fraction:
                raise ValueError("precalibration_fraction must be <= train_fraction.")

        self.fraction_training = float(train_fraction)
        self.fraction_testing = float(test_fraction)
        self.fraction_calibration = float(calibration_fraction)
        self.fraction_precal = float(precalibration_fraction)

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.basepath = os.path.dirname(__file__)
        os.makedirs(self.path_logs, exist_ok=True)

        trainset_full, testset_full = self._load_datasets()
        # self.classes = self._extract_classes(trainset_full).replace('_','')
        classes = self._extract_classes(trainset_full)

        self.classes = [cls.replace('_', '') for cls in classes]
        # self.class_to_idx = class_to_idx
        self.n_classes = len(self.classes)

        model_specification = '_'.join(f'{k}{v}' for k, v in self.augmentations.items())
        dataset_specification = (
            f'train{self.fraction_training}_test{self.fraction_testing}'
        )
        self.path_saved_model = self._build_model_path(
            model_specification=model_specification,
            dataset_specification=dataset_specification,
        )
        
        if heldout_precal:
            # Precalibration, calibration and test are three disjoint sets of the full testing data.
            assert test_fraction + calibration_fraction + precalibration_fraction <= 1, "precal + calib + test should be <= 1.0"
            calib_set_idx, testing_set_idx, precal_set_idx = UtilsDataset.generate_three_disjoint_split_idx(
                n_total=len(testset_full),
                cal_fraction=self.fraction_calibration,
                test_fraction=self.fraction_testing,
                precal_fraction=self.fraction_precal,
                seed=self.seed,
            )
            
            training_set_idx, _ = UtilsDataset.generate_set_and_subset_split_idx(
                n_total=len(trainset_full), train_fraction=self.fraction_training,
                precal_fraction=0.0, seed=self.seed
                )

        else:
            
            training_set_idx, precal_set_idx = UtilsDataset.generate_set_and_subset_split_idx(
                n_total=len(trainset_full), train_fraction=self.fraction_training,
                precal_fraction=self.fraction_precal, seed=self.seed
                )
                        
            calib_set_idx, testing_set_idx, _ = UtilsDataset.generate_three_disjoint_split_idx(
                n_total=len(testset_full),
                cal_fraction=self.fraction_calibration,
                test_fraction=self.fraction_testing,
                precal_fraction=0.0,
                seed=self.seed,
            )
            
        precalibration_source = testset_full if heldout_precal else trainset_full
        
        train_set = UtilsDataset.extract_subset_by_idx(trainset_full, subset_idx=training_set_idx.tolist())
        precalibration_set = UtilsDataset.extract_subset_by_idx(precalibration_source, subset_idx=precal_set_idx.tolist(),)
        calibration_set = UtilsDataset.extract_subset_by_idx(testset_full, subset_idx=calib_set_idx.tolist(),)
        testing_set = UtilsDataset.extract_subset_by_idx(testset_full, subset_idx=testing_set_idx.tolist(),)

        this_generator = torch.Generator().manual_seed(self.seed)

        trainloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=self.num_workers,
            pin_memory=(self.DEVICE == "cuda"),
            generator=this_generator,
        )
        # testloader = torch.utils.data.DataLoader(
        #     testing_set,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     # num_workers=self.num_workers,
        #     pin_memory=(self.DEVICE == "cuda"),
        #     generator=this_generator,
        # )
        
        self.trainset_full = trainset_full
        self.testset_full = testset_full

        print('Size of sets:')
        print(f' - Train: {len(train_set)}/{len(trainset_full)}')
        if heldout_precal:
            print(f' - Precal: {len(precalibration_set)}/{len(testset_full)} (disjoint part of testing set)')
        else:
            print(f' - Precal: {len(precalibration_set)}/{len(train_set)} (subset of training set)')
        print(f' - Test: {len(testing_set)}/{len(testset_full)} (disjoint part of testing set)')
        print(f' - Calib: {len(calibration_set)}/{len(testset_full)} (disjoint part of testing set)')

        model = self._create_model().to(self.DEVICE)
        if not self._can_evaluate_without_training(): # standard output of Resnet, then retrain last output layer
            self._load_or_train_model(model=model, trainloader=trainloader)
        
        self.model = model
        self.model.eval()

        self.calibration_data = []
        self.testing_data = []
        self.precalibration_data = []

        softmax = torch.nn.Softmax(dim=1)
        i = 1
        for input_set, output_list in [
            (calibration_set, self.calibration_data),
            (testing_set, self.testing_data),
            (precalibration_set, self.precalibration_data),
        ]:
            print(f'Loading of probability distributions ({i}/3)...')
            i += 1
            self._collect_probability_data(
                input_set=input_set,
                output_list=output_list,
                model=model,
                softmax_function=softmax,
            )

        for probDist, dataset in [(self.precalibration_data, 'Precalibration'), (self.calibration_data, 'Calibration'), (self.testing_data, 'Testing')]:
            self._evaluate_accuracies(prob_data=probDist, name_set=dataset)
            
        
        policy_name=None,
        list_other_aug=None,
        fixed_model_path=None,

    def _load_datasets(self):
        raise NotImplementedError

    def _create_model(self):
        raise NotImplementedError

    def _can_evaluate_without_training(self):
        raise NotImplementedError

    def _build_model_path(self, model_specification, dataset_specification):
        path_fixed_model = os.path.dirname(self.path_logs.rstrip("/")) + "/"
        print(path_fixed_model)
        if self.robust_training:
            return (
                f'{path_fixed_model}{self.model_file_prefix}_robust_{self.DEVICE}_'
                f'ITERATION{self.iterations}_{dataset_specification}_{model_specification}.pt'
            )

        return (
            f'{path_fixed_model}{self.model_file_prefix}_{self.DEVICE}_'
            f'ITERATION{self.iterations}_{dataset_specification}.pt'
        )
        
    def _apply_augmentation(self, data, augmentation_name):
        name = str(augmentation_name)

        if name.startswith("ROTATE_PROBE_"):
            specification = self.augmentations[augmentation_name]

            if not isinstance(specification, dict) or "degree" not in specification:
                raise ValueError(f"Missing rotation degree for augmentation {augmentation_name}.")

            return UtilsAugmentations.apply_single_augmentation(
                data,
                "ROTATE",
                rng=None,
                degree=float(specification["degree"]),
            )

        return utils.tta_policies.apply_tta_aug_policy(
            data=data,
            augmentation_name=augmentation_name,
        )

    def _load_or_train_model(self, model, trainloader):
        if os.path.isfile(self.path_saved_model):
            print('Trained model is loaded from:', self.path_saved_model)
            state_dict = torch.load(self.path_saved_model, map_location=self.DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            return

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

        for epoch in range(self.iterations):
            model.train()
            for i, (data, target) in enumerate(trainloader):
                random_aug_name = Augmentation.ORIGINAL.value
                if self.robust_training and len(self.list_augmentations) > 0:
                    random_aug_name = self.list_augmentations[random.randint(0, len(self.list_augmentations) - 1)]

                data = self._apply_augmentation(data, augmentation_name=random_aug_name)

                data = data.to(self.DEVICE)
                inputs = self._prepare_model_inputs(data)
                labels = target.to(self.DEVICE)

                optimizer.zero_grad()
                logit_out = model(inputs)
                loss = criterion(logit_out, labels)
                loss.backward()
                optimizer.step()

                if i % 100 == 0 or i == len(trainloader):
                    print(f'[Epoch: {epoch}, batch:{i}] loss: {loss.item():.3f}', flush=True)
            scheduler.step()

        torch.save(model.state_dict(), self.path_saved_model)


    def _collect_probability_data(self, input_set, output_list, model, softmax_function):
        loader = torch.utils.data.DataLoader(
            input_set,
            batch_size=self.probability_batch_size,
            shuffle=False,
            # num_workers=1,
            pin_memory=(self.DEVICE == "cuda"),
        )

        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.DEVICE, non_blocking=True)
                target_np = target.cpu().numpy()

                batch_size = data.shape[0]
                n_augmentations = len(self.list_augmentations)

                aug_batches = []
                for augmentation_name in self.list_augmentations:
                    data_aug = self._apply_augmentation(data, augmentation_name=augmentation_name)
                    aug_batches.append(data_aug)

                big_batch = torch.cat(aug_batches, dim=0)
                outputs = softmax_function(model(self._prepare_model_inputs(big_batch)))
                outputs_np = outputs.view(n_augmentations, batch_size, -1).cpu().numpy()

                for j in range(batch_size):
                    these_distributions = [
                        outputs_np[a, j, :]
                        for a in range(n_augmentations)
                    ]
                    output_list.append((these_distributions, target_np[j]))


    def _evaluate_accuracies(self, prob_data, name_set):
        correct_pred = {
            aug_name: {classname: 0 for classname in self.classes}
            for aug_name in self.list_augmentations
        }
        total_pred = {
            aug_name: {classname: 0 for classname in self.classes}
            for aug_name in self.list_augmentations
        }

        for distributions, label in prob_data:
            label = int(label)
            cls_name = self.classes[label]
            for aug_name, dist in zip(self.list_augmentations, distributions):
                if int(np.argmax(dist)) == label:
                    correct_pred[aug_name][cls_name] += 1
                total_pred[aug_name][cls_name] += 1

        self.accuracies[name_set] = []
        self.class_accuracies[name_set] = {}

        for augmentation_name in self.list_augmentations:
            # print(
            #     '==========[Accuracy on the', name_set,
            #     'dataset (Augmentation:', augmentation_name, ')]=========='
            # )

            self.class_accuracies[name_set][augmentation_name] = []

            for cls_name in self.classes:
                # cls_name = cls_name.replace('_', '\\_')
                c = correct_pred[augmentation_name][cls_name]
                t = total_pred[augmentation_name][cls_name]
                accuracy = round(100.0 * c / t, 2) if t > 0 else 0.0

                # print('Class', cls_name, ':', accuracy, '% (', t, 'cases)')
                self.class_accuracies[name_set][augmentation_name].append((cls_name, accuracy, t))

            total_correct = sum(correct_pred[augmentation_name].values())
            total_cases = sum(total_pred[augmentation_name].values())
            overall = round(100.0 * total_correct / total_cases, 2) if total_cases > 0 else 0.0

            self.accuracies[name_set].append(overall)
            print('Overall accuracy:', overall, '% (', total_cases, 'cases)', ' ', augmentation_name)

        print('\n', flush=True)

    def _prepare_model_inputs(self, data):
        raise NotImplementedError


    def _texClassAccuracies(self, dataset: str) -> str:
        data = self.class_accuracies.get(dataset, {})
        augs = list(self.list_augmentations)

        acc_map = {aug: {} for aug in augs}
        for aug in augs:
            for cls, acc, n in data.get(aug, []):
                acc_map[aug][cls] = (float(acc), int(n))

        classes = list(getattr(self, 'classes', []))
        if not classes:
            classes = sorted({cls for aug in augs for cls, _, _ in data.get(aug, [])})

        colspec = 'l|r|' + '|'.join(['r'] * len(augs))

        header = []
        header.append(f'\\begin{{tabular}}{{{colspec}}}')
        header.append('\\hline')
        header.append(
            '\\textbf{Class} & \\textbf{N} & '
            + f'\\multicolumn{{{len(augs)}}}{{c}}{{\\textbf{{Accuracies}}}} \\\\ \\hline'
        )
        header.append(
            ' &  & ' + ' & '.join([f'\\textbf{{{aug}}}' for aug in augs]) + ' \\\\ \\hline'
        )

        rows = []
        for cls in classes:
            n_val = ''
            for aug in augs:
                if cls in acc_map[aug]:
                    n_val = str(acc_map[aug][cls][1])
                    break

            acc_vals = []
            for aug in augs:
                if cls in acc_map[aug]:
                    acc_vals.append(f'{acc_map[aug][cls][0]:.2f}\\%')
                else:
                    acc_vals.append('--')

            rows.append(f'{cls} & {n_val} & ' + ' & '.join(acc_vals) + ' \\\\ \\hline')

        if not rows:
            rows = [f'\\multicolumn{{{2 + len(augs)}}}{{c}}{{No class accuracies available}} \\\\ \\hline']

        return '\n'.join(header + rows + ['\\end{tabular}'])

    def texInfo(self):
        # text1 = """
        # \\begin{{tabular}}{{l|l}} \\
        # \\textbf{{Benchmark name:}} & {0} \\\\ \\hline
        # \\textbf{{Augmentations:}} & {1} \\\\ \\hline
        # # \\textbf{{Accuracy Precalibration set:}} & {2} \\% \\\\ \\hline
        # \\textbf{{Accuracy Testing set:}} & {3} \\% \\
        # \\end{{tabular}}
        # """.format(
        #     self.benchmark_name,
        #     list(self.augmentations.values()),
        #     # self.accuracies['Training'],
        #     self.accuracies['Precalibration'],
        #     # self.accuracies['Calibration'],
        #     self.accuracies['Testing'],
        # )

        # final = """
        # \\section{Accuracies Per Class on Training Set}
        # """
        # final += self._texClassAccuracies('Precalibration')

        # final += """
        # \\newpage
        # \\section{Accuracies Per Class on Test Set}
        # """

        # final += self._texClassAccuracies('Testing')
        # return text1 + final
        return """This part has been removed for the moment."""

    @staticmethod
    def _extract_classes(dataset):
        classes = getattr(dataset, 'classes', None)
        if classes is None:
            raise ValueError('Dataset does not expose a classes attribute.')

        normalized_classes = []
        for class_name in classes:
            if isinstance(class_name, (tuple, list)):
                normalized_classes.append(', '.join(str(item) for item in class_name))
            else:
                normalized_classes.append(str(class_name))

        return tuple(normalized_classes)
    