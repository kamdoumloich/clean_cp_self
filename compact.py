#!/usr/bin/env python3
import os, math

import numpy as np

from evaluation_metrics import compute_all_metrics, metrics_to_tex, metrics_comparison_table


def runCompleteEvaluation(texTargetFile,benchmark,conformanceApproaches,interestingCaseFilter,alpha=None):
    import math
    def sizeY(s):
        return 5.0*math.log1p(s)/math.log1p(len(benchmark.classes))
    
    os.makedirs(os.path.dirname(texTargetFile), exist_ok=True)
    with open(texTargetFile,"w") as outFile:
        
        # Prefix of the evaluation
        outFile.write("""\\documentclass[parskip=half]{scrartcl}
        \\usepackage{tikz}
        \\usepackage{pxfonts}
        \\usepackage[a4paper,left=1cm,right=1cm,top=2cm,bottom=3cm]{geometry}
        \\title{Conformance Testing Report}
        \\author{\\relax}\\date{\\today}
        \\begin{document} \\maketitle
        \\section{Benchmark information}""")
        outFile.write(benchmark.texInfo())

        outFile.write("""\\section{{Predictor Information}}""")
        for i in range(0,len(conformanceApproaches)):
            outFile.write("""\\subsection{{Conformance Predictor {0} }}""".format(i))
            outFile.write(conformanceApproaches[i].texInfo())

        # Compare different conformance predictors.
        # 1. Prepare graphs
        outFile.write("\\section{Direct comparison between all conformance prediction approaches}\n")
        outFile.write("The following comparison shows the performance of different conformance predictors. The X Axis shows the needed precision, while the Y axis shows, for each conformance prediction approach, precision on the conformance calibration set and the resulting mean conformance set sizes.\n")
        yscale=1.0
        if len(conformanceApproaches)>=2:
            yscale = 2.5/len(conformanceApproaches)
        outFile.write("\n\\begin{tikzpicture}[yscale="+str(yscale)+"]")
        outFile.write("\\draw (0,0) rectangle (10,"+str(10*len(conformanceApproaches))+");\n")

        # 2. Draw boundaries
        for i in range(0,len(conformanceApproaches)):
            outFile.write("\\draw (0,"+str(10*i+5)+") -- +(10,0);")
            if (i>=0) and i<len(conformanceApproaches)-1:
                outFile.write("\\draw (0,"+str(10*i+10)+") -- +(10,0);")
            outFile.write("\\node[anchor=east] at (-0.25,"+str(10*i+0+2.5)+") {\\rotatebox{90}{\\scriptsize \\textsf{ "+conformanceApproaches[i].getShortName()+", Accuracy}}};\n")
            outFile.write("\\node[anchor=east] at (0,"+str(10*i+5+2.5)+") {\\rotatebox{90}{\\scriptsize \\textsf{ "+conformanceApproaches[i].getShortName()+", Set sizes}}};\n")

        # 3. Draw diagonal line percentage
        for conformanceAlgoNo in range(0,len(conformanceApproaches)):
            outFile.write("\\draw[color=black!50!white] (0,"+str(10*conformanceAlgoNo)+") -- +(10,5);\n")

        # 4. Define translation functions for coordinates
        def getPercentageCoord(val):
            val = min(1,max(0,val))
            val = 1-val
            val += 0.01
            return 10.0 - 10.0*(math.log(val)-math.log(0.01))/(math.log(1.01)-math.log(0.01))
        def getPercentageCoordY(val):
            val = min(1,max(0,val))
            val = 1-val
            val += 0.01
            return 5.0 - 5.0*(math.log(val)-math.log(0.01))/(math.log(1.01)-math.log(0.01))

        # 4. Draw graph labeling
        for ylabel in [0.25,0.5,0.7,0.8,0.9,0.95,0.98,0.99,0.9975]:
            outFile.write("\\draw[color=black!50!white] ("+str(getPercentageCoord(ylabel))+",0) -- +(0,-0.2) node[below] {\\scriptsize "+str(ylabel)+"};\n")
            outFile.write("\\draw[color=black!50!white,dashed] ("+str(getPercentageCoord(ylabel))+",0) -- +(0,"+str(len(conformanceApproaches)*10)+");\n")
        for conformanceAlgoNo in range(0,len(conformanceApproaches)):
            for xlabel in [0.25,0.5,0.7,0.8,0.9,0.95,0.98,0.99,0.9975]:
                outFile.write("\\draw[color=black!50!white] (10,"+str(getPercentageCoordY(xlabel)+conformanceAlgoNo*10)+") -- +(0.2,0) node[right] {\\scriptsize "+str(xlabel)+"};\n")
            K = len(benchmark.classes)
            # step = max(1, K // 8)
            # for x in range(step, K, step):
            for x in (1,2,5,10,20,50,100,250,1000):
                # yy = conformanceAlgoNo*10 + 5 + 5.0*x/K
                yy = conformanceAlgoNo*10 + 5 + sizeY(x)
                outFile.write(f"\\draw[color=black!50!white,dashed] (0,{yy}) -- +(10,0);\n")
                outFile.write(f"\\draw[color=black!50!white] (10,{yy}) -- +(0.2,0) node[right] {{\\scriptsize {x}}};\n")
            
        eval_alpha = alpha if alpha is not None else (1.0 - interestingCaseFilter.probabilityOfInterest)

        for conformanceAlgoNo in range(0, len(conformanceApproaches)):
            predictor = conformanceApproaches[conformanceAlgoNo]
            print("\n\nCase:", predictor.getShortName())
            print("Precalibration started...")
            predictor.precalibrate()                       # learn theta once
            print("Precalibration done.")
            
            # currentParameter = 0.0
            # data = []
            # done = False
            
            # while not done:
            #     correctCases = 0 
            #     sumSizes = 0
            #     nextParameter = 1.0
            #     allCases = 0
            #     interestingCaseFilter.reportNewContext(conformanceAlgoNo,currentParameter)
            #     for i,(probabilities,classification) in enumerate(benchmark.calibration_data):
            #         # print("USE PROB: ",probabilities)
            #         # classes,nextUp = conformanceApproaches[conformanceAlgoNo].predict(probabilities,currentParameter)
            #         classes,nextUp = algoOject.predict(probabilities,currentParameter)
            #         sumSizes += len(classes)
            #         allCases += 1
            #         if classification in classes:
            #             # Correct!
            #             correctCases +=1
            #         else:
            #             nextParameter = min(nextUp,nextParameter)
            #         interestingCaseFilter.reportCase(i,probabilities,classification,classes)
            #     percentage = correctCases/allCases
            #     print("Current Param",currentParameter," correct: ",correctCases,allCases)
            #     print("Next parameter: ",nextParameter)
            #     interestingCaseFilter.reportProbabilityOfBeingCorrect(percentage)
            #     # Compute Alpha from the percentage
            #     realLevel = percentage*allCases/(allCases+1)
            #     data.append((currentParameter,realLevel,sumSizes/allCases))

            #     if correctCases==allCases:
            #         done = True
            #     else:
            #         assert currentParameter!=nextParameter # Finds bugs in the approaches
            #         currentParameter=nextParameter

        #     # ---- alpha-grid: one calibration per alpha ----
        #     alpha_grid = sorted(
        #         # {0.2, 0.10, 0.05, 0.02, 0.01, 0.005},
        #         # {0.2, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0},
        #         {0.001},
        #         reverse=True
        #     )

        #     data = []

        #     for a_g in alpha_grid:
        #         print(f"Evaluating alpha={a_g}")

        #         _ = predictor.calibrate(
        #             benchmark.calibration_data,
        #             alpha=float(a_g)
        #         )

        #         correct = 0
        #         total = 0
        #         sizes = 0

        #         for probabilities, classification in benchmark.calibration_data:
        #             classes, _ = predictor.predict(probabilities, )

        #             sizes += len(classes)
        #             total += 1
        #             correct += int(classification in classes)

        #         coverage_g = correct / total
        #         mean_size_g = sizes / total

        #         print(
        #             f"alpha={a_g:.4f}, "
        #             f"coverage={coverage_g:.4f}, "
        #             f"mean_size={mean_size_g:.4f}"
        #         )

        #         data.append(
        #             (a_g, coverage_g, mean_size_g)
        #         )

        #     # data.sort(key=lambda t: t[1])
        #     data.sort(key=lambda t: 1.0 - t[0])

        #     # Draw corresponding curves
        #     # a) Real probability of the conformance predictor being correct
        #     outFile.write("\\draw[color=red,thick] (0,"+str(conformanceAlgoNo*10)+")")
        #     for a_g, level, meanSize in data:
        #         target_coverage = 1.0 - a_g

        #         outFile.write(" |- ")
        #         outFile.write(
        #             "("
        #             + str(getPercentageCoord(target_coverage)) + ","
        #             + str(getPercentageCoordY(level) + conformanceAlgoNo * 10)
        #             + ")"
        #         )
            
        #     outFile.write(";\n")
        #     # b) Mean sizes of the conformance sets
        #     outFile.write("\\draw[color=blue,thick] (0,"+str(conformanceAlgoNo*10+5)+")")
        #     for a_g, level, meanSize in data:
        #         target_coverage = 1.0 - a_g

        #         outFile.write(" |- ")
        #         outFile.write(
        #             "("
        #             + str(getPercentageCoord(target_coverage))  + ","
        #             + str(sizeY(meanSize) + 5 + conformanceAlgoNo * 10)
        #             + ")"
        #         )
        #     outFile.write(";\n")

            # ---- leave the predictor at the evaluation operating point for the test split ----
            qhat_eval = predictor.calibrate(benchmark.calibration_data, eval_alpha)   # sets predictor.qhat
            interestingCaseFilter.reportNewContext(conformanceAlgoNo, qhat_eval)


        # X. End Comparison
        outFile.write("\\end{tikzpicture}")

        # X+1 Interesting cases found
        outFile.write("\\section{Interesting cases found Calibration}\n")
        outFile.write(interestingCaseFilter.getTexReport())

        # Testing
        outFile.write("\\section{Performance on test data}\n")

        K = len(benchmark.classes)
        interestingCaseFilter.resetFilter()

        all_metrics = []
        # alpha for size-stratified target line; fall back to realized coverage if not given
        _alpha_for_metrics = alpha if alpha is not None else None
        
        np.savetxt("benchmark_testing_data.log", benchmark.testing_data[0][0])

        for conformanceAlgoNo in range(len(conformanceApproaches)):
            correctCases = 0
            sumSizes = 0
            allCases = 0
            # calibrationParameter = listCalibration[conformanceAlgoNo]

            n_set_of_size = np.zeros(K + 1, dtype=int)  # 0..K
            rec_sizes = []
            rec_covered = []
            rec_ytrue = []
            rec_difficulty = []  # input-side: 1 - max(p_orig), shared across methods

            predictor = conformanceApproaches[conformanceAlgoNo]
            
            from collections import defaultdict

            cell_stats = defaultdict(lambda: {
                "n": 0,
                "top1_correct": 0,
                "covered": 0,
                "sum_size": 0,
                "qhat": None,
                "true_scores": [],
            })
            
            with open(f"test_stat_{predictor.getShortName()}.log", 'w') as f:
                f.write("Predicted set    | True class\n")
                
            # print(np.array([benchmark.testing_data]).shape)

            # for i, (probabilities, classification) in enumerate(benchmark.testing_data):
            #     classes, _ = predictor.predict(probabilities)

            probabilities, targets = benchmark.testing_data

            for i, classification in enumerate(targets):
                these_probabilities = probabilities[:, i, :]

                classes, _ = predictor.predict(these_probabilities)
    
                # with open(f"test_stat_{predictor.getShortName()}.log", 'a') as f:
                #     f.write(f"{classification} -- {classes}\n")
                
                #----------------------------------
                if hasattr(predictor, "last_cell") and predictor.last_cell is not None:
                    c = int(predictor.last_cell)
                    s = cell_stats[c]

                    s["n"] += 1
                    s["top1_correct"] += int(
                        np.argmax(these_probabilities[0]) == classification
                    )
                    s["covered"] += int(classification in classes)
                    s["sum_size"] += len(classes)
                    s["qhat"] = float(predictor.last_qhat)

                    s["true_scores"].append(
                        float(predictor.last_scores[classification])
                    )
                #0000000000000000000000000000000000
    
                s = len(classes)

                sumSizes += s
                allCases += 1

                if s <= K:
                    n_set_of_size[s] += 1
                else:
                    # should never happen; indicates a bug in predict()
                    raise ValueError(f"Predicted set size {s} > number of classes {K}")

                is_cov = 1 if classification in classes else 0
                if is_cov:
                    correctCases += 1

                rec_sizes.append(s)
                rec_covered.append(is_cov)
                rec_ytrue.append(classification)
                
                _p_orig = np.asarray(these_probabilities[0], dtype=np.float64)

                if predictor.inputs_type == "logits":
                    z = _p_orig - np.max(_p_orig)
                    p_orig = np.exp(z)
                    p_orig /= p_orig.sum()

                elif predictor.inputs_type == "probs":
                    p_orig = np.clip(_p_orig, 0.0, None)
                    p_orig /= p_orig.sum()

                else:
                    raise ValueError(f"Unknown inputs_type={predictor.inputs_type}")

                difficulty = 1.0 - float(np.max(p_orig))

                rec_difficulty.append(difficulty)

                interestingCaseFilter.reportCase(i, probabilities, classification, classes)

            if allCases == 0:
                outFile.write(f"\\paragraph{{Predictor:{predictor.getShortName()}}} No test cases.\\\\\n")
                continue

            coverage = correctCases / allCases
            avg_size = sumSizes / allCases

            outFile.write(
                f"\\paragraph{{Predictor:{predictor.getShortName()}}} "
                f"calibration at alpha={eval_alpha}. "
                f"calibration value={predictor.qhat}. "
                f"Correct: {correctCases}/{allCases}\\\\\n"
            )
            outFile.write(f"Average set size: {avg_size}   Coverage: {coverage}\\\\\n")
            outFile.write("Set size histogram (0..K):\\\\\n")
            outFile.write(" | ".join(map(str, n_set_of_size.tolist())))
            outFile.write("\\\\\n")
            
            # print("\n=== Per-cell diagnostics ===")

            # for c in sorted(cell_stats):
            #     s = cell_stats[c]
            #     scores = np.asarray(s["true_scores"])

            #     print(
            #         f"Cell {c}: "
            #         f"n={s['n']}, "
            #         f"qhat={s['qhat']:.6f}, "
            #         f"top1_acc={s['top1_correct']/s['n']:.4f}, "
            #         f"coverage={s['covered']/s['n']:.4f}, "
            #         f"avg_size={s['sum_size']/s['n']:.3f}, "
            #         f"score_median={np.median(scores):.6f}, "
            #         f"score_q90={np.quantile(scores, 0.90):.6f}, "
            #         f"score_q99={np.quantile(scores, 0.99):.6f}"
            #     )

            metrics = compute_all_metrics(
                sizes=rec_sizes,
                covered=rec_covered,
                y_true=rec_ytrue,
                difficulty=rec_difficulty,
                alpha=(_alpha_for_metrics if _alpha_for_metrics is not None else (1.0 - coverage)),
                n_classes=K,
            )
            
            outFile.write(metrics_to_tex(metrics, predictor.getShortName()))
            all_metrics.append((predictor.getShortName(), metrics))

            interestingCaseFilter.reportProbabilityOfBeingCorrect(coverage)

        # Comparison table across all predictors
        outFile.write("\\subsection{Metric comparison across predictors}\n")
        outFile.write(metrics_comparison_table(all_metrics))

        # X+1 Interesting cases found
        outFile.write("\\section{Interesting cases found Testing}\n")
        outFile.write(interestingCaseFilter.getTexReport());

        # End
        outFile.write("""
        \\end{document}
        """)