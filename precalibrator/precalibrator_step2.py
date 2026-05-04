#!/usr/bin/env python3
import argparse
import bisect
import multiprocessing as mp
import os
import sys, math, copy


_WORKER_PATTERNS = None
_WORKER_INTERVAL_START_INDEX = None
_WORKER_NOF_SAMPLES_NEEDED = None


def _default_jobs():
    env_value = os.environ.get("PRECALIBRATOR_STEP2_JOBS")
    if env_value is not None:
        return max(1, int(env_value))
    return max(1, min(os.cpu_count() or 1, 8))


def _set_worker_context(patterns, interval_start_index, nof_samples_needed):
    global _WORKER_PATTERNS
    global _WORKER_INTERVAL_START_INDEX
    global _WORKER_NOF_SAMPLES_NEEDED

    _WORKER_PATTERNS = patterns
    _WORKER_INTERVAL_START_INDEX = interval_start_index
    _WORKER_NOF_SAMPLES_NEEDED = nof_samples_needed


def _build_interval_start_index(patterns):
    interval_starts = []
    for pattern in patterns:
        pattern_starts = []
        for intervals in pattern:
            pattern_starts.append([interval[0] for interval in intervals])
        interval_starts.append(pattern_starts)
    return interval_starts


def _find_interval_size_for_calibration(intervals, interval_starts, calibrationValue):
    if not intervals:
        return None

    candidate_pos = bisect.bisect_right(interval_starts, calibrationValue) - 1
    if candidate_pos < 0:
        return None

    start, end, size = intervals[candidate_pos]
    if start < calibrationValue and end > calibrationValue:
        return size
    return None


def _evaluate_calibration_candidate(task):
    candidateNo, calibrationValue = task
    patterns = _WORKER_PATTERNS
    interval_start_index = _WORKER_INTERVAL_START_INDEX
    nofSamplesNeeded = _WORKER_NOF_SAMPLES_NEEDED

    currentSelection = [0 for patternNo in range(0, len(patterns))]
    currentSizesConformanceSets = []
    currentSize = 0

    for patternNo in range(0, len(patterns)):
        size = _find_interval_size_for_calibration(
            patterns[patternNo][0],
            interval_start_index[patternNo][0],
            calibrationValue,
        )
        if size is None:
            return candidateNo, calibrationValue, False, math.inf, None, currentSize
        currentSizesConformanceSets.append(size)

    while currentSize < nofSamplesNeeded:
        bestRatio = None
        bestPattern = None
        bestNofAddCoverage = None
        bestNofAddConformanceSets = None

        for patternNo in range(0, len(patterns)):
            for target in range(currentSelection[patternNo] + 1, len(patterns[patternNo])):
                size = _find_interval_size_for_calibration(
                    patterns[patternNo][target],
                    interval_start_index[patternNo][target],
                    calibrationValue,
                )
                if size is None:
                    continue

                nextDelta = size - currentSizesConformanceSets[patternNo]
                nextAddition = min(
                    target - currentSelection[patternNo],
                    nofSamplesNeeded - currentSize,
                )
                nextRatio = nextDelta / float(nextAddition)
                if (bestPattern is None) or nextRatio < bestRatio:
                    bestRatio = nextRatio
                    bestPattern = patternNo
                    bestNofAddCoverage = nextAddition
                    bestNofAddConformanceSets = nextDelta

        if bestNofAddCoverage is None:
            return candidateNo, calibrationValue, False, math.inf, None, currentSize

        currentSizesConformanceSets[bestPattern] += bestNofAddConformanceSets
        currentSelection[bestPattern] += bestNofAddCoverage
        currentSize += bestNofAddCoverage

    return (
        candidateNo,
        calibrationValue,
        True,
        sum(currentSizesConformanceSets),
        currentSelection,
        currentSize,
    )


def _evaluate_candidates(tasks, patterns, interval_start_index, nofSamplesNeeded, jobs, progress_every):
    totalCalibrationValues = len(tasks)
    jobs = max(1, min(int(jobs), totalCalibrationValues))
    _set_worker_context(patterns, interval_start_index, nofSamplesNeeded)

    if jobs == 1:
        results = []
        for candidateNo, task in enumerate(tasks, start=1):
            if candidateNo == 1 or candidateNo == totalCalibrationValues or candidateNo % progress_every == 0:
                print(
                    f"[step2] Candidate {candidateNo}/{totalCalibrationValues}: {task[1]}",
                    flush=True,
                )
            results.append(_evaluate_calibration_candidate(task))
        return results

    chunksize = max(1, totalCalibrationValues // (jobs * 8))
    print(f"[step2] Running candidate evaluation with {jobs} workers.", flush=True)
    results = []
    with mp.Pool(
        processes=jobs,
        initializer=_set_worker_context,
        initargs=(patterns, interval_start_index, nofSamplesNeeded),
    ) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(_evaluate_calibration_candidate, tasks, chunksize=chunksize),
            start=1,
        ):
            if completed == 1 or completed == totalCalibrationValues or completed % progress_every == 0:
                print(
                    f"[step2] Completed {completed}/{totalCalibrationValues} candidates.",
                    flush=True,
                )
            results.append(result)

    return results


def performCalibrationStep2(inputFile,percentage,outputFile, progress_every=100, jobs=None):

    patterns = []

    with open(inputFile,"r") as inputFile:
        # First line - Number of patterns
        firstLine = inputFile.readline().strip().split(" ")
        assert firstLine[0]=="#Patterns:"
        assert len(firstLine)==2
        nofPatterns = int(firstLine[1])
        for pattern in range(nofPatterns):
            thisPattern = []
            nextLine = inputFile.readline().strip().split(" ")
            assert nextLine[0]=="#:"
            assert len(nextLine)==2
            nofElementsInPatternMinusOne = int(nextLine[1])
            for element in range(nofElementsInPatternMinusOne):
                nextLine=inputFile.readline().strip().split(" ")
                assert nextLine[0]=="##:"
                assert len(nextLine)==2
                sequence = []
                nofSequenceElements = int(nextLine[1])
                for i in range(nofSequenceElements):
                    nextLine=inputFile.readline().strip().split(" ")
                    assert len(nextLine)==3
                    nextLine = (float(nextLine[0]),float(nextLine[1]),float(nextLine[2]))
                    sequence.append(nextLine)
                thisPattern.append(sequence)
            patterns.append(thisPattern)

    # Patterns is now filled
    del pattern
    nofSamplesOverall = sum([len(a) for a in patterns])-sum([1 for a in patterns])
    print("Number of samples overall:",nofSamplesOverall)
    nofSamplesNeeded = math.ceil(nofSamplesOverall*percentage)
    print("Number of samples needed:",nofSamplesNeeded)

    possibleCalibrationValues = set([])
    for a in patterns: # Patterns
        for b in a: # Number of correct guesses
            for c in b: # Element in the intervals
                possibleCalibrationValues.add(c[0])
                possibleCalibrationValues.add(c[1])
    if len(possibleCalibrationValues)==0:
        raise ValueError("No feasible calibration intervals were produced by precalibrator step 1.")
    possibleCalibrationValues = list(possibleCalibrationValues)
    possibleCalibrationValues.sort()
    if possibleCalibrationValues[0]>0.0:
        possibleCalibrationValues = [0.0]+possibleCalibrationValues
    if possibleCalibrationValues[-1]<1.0:
        possibleCalibrationValues = possibleCalibrationValues+[1.0]
    calibrationValuesToTry = []
    for i in range(len(possibleCalibrationValues)-1):
        middle = 0.5*possibleCalibrationValues[i]+0.5*possibleCalibrationValues[i+1]
        calibrationValuesToTry.append(middle)
    # print("Calibration Values to try: ",possibleCalibrationValues)

    interval_start_index = _build_interval_start_index(patterns)

    totalCalibrationValues = len(calibrationValuesToTry)
    print(
        "[step2] Evaluating",
        totalCalibrationValues,
        "candidate calibration values across",
        len(patterns),
        "pattern groups.",
        flush=True,
    )

    bestCalibrationValue = math.inf
    bestSizeConformanceSets = math.inf
    bestSelection = None
    jobs = _default_jobs() if jobs is None else int(jobs)
    tasks = list(enumerate(calibrationValuesToTry, start=1))

    results = _evaluate_candidates(
        tasks=tasks,
        patterns=patterns,
        interval_start_index=interval_start_index,
        nofSamplesNeeded=nofSamplesNeeded,
        jobs=jobs,
        progress_every=max(1, int(progress_every)),
    )

    for candidateNo, calibrationValue, feasible, sizeConformanceSets, selection, currentSize in sorted(results):
        if feasible and bestSizeConformanceSets >= sizeConformanceSets:
            bestCalibrationValue = calibrationValue
            bestSizeConformanceSets = sizeConformanceSets
            bestSelection = selection



    print("Final calibration value:",bestCalibrationValue,"with conformance sets sizes",bestSizeConformanceSets)
    if bestSelection is None or not math.isfinite(bestCalibrationValue):
        raise ValueError("Could not determine a feasible global calibration value in precalibrator step 2.")

    with open(outputFile,"w") as resultFile:
        resultFile.write(str(bestCalibrationValue)+"\n")
        for a in bestSelection:
            resultFile.write(str(int(a))+"\n")





def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="?", default="out-short-per-class.txt")
    parser.add_argument("percentage", nargs="?", type=float, default=0.9)
    parser.add_argument(
        "output_file",
        nargs="?",
        default="out-short-per-class-input-to-step-3.txt",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print step-2 progress every N calibration candidates.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=_default_jobs(),
        help=(
            "Number of parallel worker processes for candidate evaluation. "
            "Can also be set with PRECALIBRATOR_STEP2_JOBS."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    performCalibrationStep2(
        args.input_file,
        args.percentage,
        args.output_file,
        progress_every=max(1, int(args.progress_every)),
        jobs=max(1, int(args.jobs)),
    )
