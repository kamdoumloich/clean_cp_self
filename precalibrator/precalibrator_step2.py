#!/usr/bin/env python3
import argparse
import sys, math, copy

def performCalibrationStep2(inputFile,percentage,outputFile):

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

    bestCalibrationValue = math.inf
    bestSizeConformanceSets = math.inf
    bestSelection = None

    for calibrationValue in calibrationValuesToTry:
        print("Trying calibration value:",calibrationValue)

        # Build initial selection
        currentSelection = [0 for patternNo in range(0,len(patterns))]
        currentSizesConformanceSets = []
        currentSize = 0

        failedToFindAllInitialValues = False
        for patternNo in range(0,len(patterns)):
            # Search for
            foundOne = False
            for (a,b,c) in patterns[patternNo][0]:
                if (a<calibrationValue) and (b>calibrationValue):
                    foundOne = True
                    currentSizesConformanceSets.append(c)
            if not foundOne:
                failedToFindAllInitialValues = True
            currentSize = 0

        if not failedToFindAllInitialValues:
            # print("Initial size:",currentSize)

            # Run greedy algorithm
            done = False
            while currentSize<nofSamplesNeeded:
                bestRatio = None
                bestPattern = None
                bestNofAddCoverage = None
                bestNofAddConformanceSets = None
                for patternNo in range(0,len(patterns)):
                    for target in range(currentSelection[patternNo]+1,len(patterns[patternNo])):
                        # print("Trying:",patternNo,target)
                        for (a,b,c) in patterns[patternNo][target]:
                            if (a<calibrationValue) and (b>calibrationValue):
                                nextDelta = c-currentSizesConformanceSets[patternNo]
                                nextAddition = min(target-currentSelection[patternNo],nofSamplesNeeded-currentSize)
                                nextRatio = nextDelta/float(nextAddition)
                                if (bestPattern is None) or nextRatio<bestRatio:
                                    bestRatio = nextRatio
                                    bestPattern = patternNo
                                    bestNofAddCoverage = nextAddition
                                    bestNofAddConformanceSets = nextDelta
                if bestNofAddCoverage is None:
                    # Can't achieve needed coverage.
                    print("Cannot achieve coverage! Number of correct values achievable:",currentSize)
                    currentSize = math.inf
                    currentSizesConformanceSets[0] = math.inf
                else:
                    # print("Best: ",bestRatio,bestNofAddCoverage,bestNofAddConformanceSets,bestPattern,currentSelection[bestPattern])
                    currentSizesConformanceSets[bestPattern] += bestNofAddConformanceSets
                    currentSelection[bestPattern] += bestNofAddCoverage
                    currentSize += bestNofAddCoverage
            print("Final result with",currentSize," coverage and ",sum(currentSizesConformanceSets)," big conformance sets overall.")
            if bestSizeConformanceSets>=sum(currentSizesConformanceSets):
                bestCalibrationValue = calibrationValue
                bestSizeConformanceSets = sum(currentSizesConformanceSets)
                bestSelection = currentSelection



    print("Final calibration value:",bestCalibrationValue,"with conformance sets sizes",bestSizeConformanceSets)

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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    performCalibrationStep2(args.input_file, args.percentage, args.output_file)
