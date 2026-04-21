#!/usr/bin/env python3
import sys, math, copy
import pyscipopt 
import multiset

import faulthandler
import signal
import sys

def dump_stack(sig, frame):
    faulthandler.dump_traceback(file=sys.stderr)

signal.signal(signal.SIGINT, dump_stack)

def processPatternSimplified(patternData):

    # Simplified approach without linear programming (for big patterns)
    nofClasses = (len(patternData[0])-1)//2
    finalOutput = []

    for numberOfCorrectClassifications in range(0,len(patternData)+1):

        sys.stdout.write("(proc:"+str(numberOfCorrectClassifications)+")")
        sys.stdout.flush()

        weightlayers = []
        for weight in [0.0,0.25,0.5,0.75,1.0]:
            allIntervalsForThePattern = []

            # Mix according to the weight
            theseMixes = []
            for pattern in patternData:
                thisMix = []
                for i in range(nofClasses):
                    thisMix.append((1-weight)*pattern[i] + weight*pattern[i+nofClasses])
                thisMix.append(int(pattern[-1]))
                theseMixes.append(thisMix)

            # Which calibrationValues to consider
            allCalibrationValuesToConsider = multiset.Multiset([0.0])
            correctClassCalibrationValuesToConsider = multiset.Multiset([])
            for mix in theseMixes:
                allCalibrationValuesToConsider.update(mix)
                correctClassCalibrationValuesToConsider.add(mix[mix[-1]])
            allCalibrationValuesToConsider = list(allCalibrationValuesToConsider)
            allCalibrationValuesToConsider.sort()
            correctClassCalibrationValuesToConsider = list(correctClassCalibrationValuesToConsider)
            correctClassCalibrationValuesToConsider.sort()

            # Now build intervals
            thisIntervalStart = 0.0
            thisIntervalQuality = nofClasses*len(patternData)
            theseIntervals = []
            currentPosCorrect = len(patternData)
            for i in range(1,len(allCalibrationValuesToConsider)):
                nextPoint = allCalibrationValuesToConsider[i]
                previousPoint = allCalibrationValuesToConsider[i-1]

                while currentPosCorrect>0 and previousPoint>=correctClassCalibrationValuesToConsider[len(patternData)-currentPosCorrect]:
                    currentPosCorrect -= 1
                
                if currentPosCorrect>=numberOfCorrectClassifications:
                    theseIntervals.append((previousPoint,nextPoint,currentPosCorrect))

            weightlayers.append(theseIntervals)

        sys.stdout.write("(mix:"+str(len(weightlayers[0]))+")")
        sys.stdout.flush()

        # Postprocess weight layers to single intervals
        positionPoints = set([])
        for a in weightlayers:
            for (b,c,d) in a:
                positionPoints.add(b)    
                positionPoints.add(c)
        positionPoints = list(positionPoints)
        positionPoints.sort()
        assert positionPoints[0]==0.0
        # 1. Make long list before merging
        allIntervals = []
        for i in range(len(positionPoints)-1):
            bestValue = math.inf
            start = positionPoints[i]
            end = positionPoints[i+1]
            for a in weightlayers:
                minPos = 0
                maxPos = len(a)
                while (maxPos!=minPos):
                    middlePos = (minPos+maxPos)//2
                    (b,c,d) = a[middlePos]
                    if (b<=start):
                        if (c >= end):
                            bestValue = min(bestValue,d)
                            maxPos = minPos
                        else:
                            minPos = middlePos+1
                    else:
                        maxPos = middlePos
            if not bestValue==math.inf:
                allIntervals.append((start,end,bestValue))

        # 2. Merge intervals
        filteredIntervals = []
        currentStart = None
        currentValue = None
        lastEnd = None
        for (b,c,d) in allIntervals:
            if currentStart is None:
                currentStart = b
                currentValue = d
                lastEnd = c
            elif (currentValue!=d) or (b!=lastEnd):
                filteredIntervals.append((currentStart,lastEnd,currentValue))
                currentStart = b
                currentValue = d
                lastEnd = c
            else:
                lastEnd = c
        if not currentStart is None:
            filteredIntervals.append((currentStart,lastEnd,currentValue))

        finalOutput.append(filteredIntervals)
        

    print("Merged intervals:")
    for a in finalOutput:
        print(a)


    return finalOutput




def processPattern(patternData):
    """Process individual pattern."""
    epsilon = 0.00001
    allQualityIntervals = []
    lastRealCoverage = None
    for numberOfCorrectClassifications in range(0,len(patternData)+1):
        # print("Trying to find best possible values # correct:",numberOfCorrectClassifications)
        assert (len(patternData[0]) & 1) != 0
        nofClasses = (len(patternData[0])-1)//2

        # Caching: The last solution was actually uniformly better than needed
        if (not lastRealCoverage is None) and lastRealCoverage>=numberOfCorrectClassifications:
            thisCopy = copy.copy(allQualityIntervals[-1])
            allQualityIntervals.append(thisCopy)
        else:

            # Approach: 
            # 0. Iterate over the numbers of correct classifications
            # 1. Successively go through the intervals of calibration value intervals.
            # 2. For each interval:
            #   2.1. Solve an MILP process trying to find a weight value minimizing the sum of the conformance set for some calibration value in the the set
            #   2.2  Solve an LP that for the combination of concrete conformance set finds which calibration values a weight value can be found for
            #   2.3. Update all intervals to remove the covered space.

            # Try to find the achievable combinations of calibration value and sizes of the belief sets
            
            thisQualityIntervals = set([])
            theseRealCoverage = set([])
            
            remainingCalibrationValueIntervals = [(0.0,1.0)]
            while len(remainingCalibrationValueIntervals)>0:
                (thisStart,thisEnd) = remainingCalibrationValueIntervals[0]

                # Build MILP instance
                model = pyscipopt.Model("findset")
                model.hideOutput(True)
                z = {}
                weight = model.addVar(name="weight", vtype="C", lb=0.0, ub=1.0)
                # Use epsilons in the following bounds in order to ensure that the same solution is not found twice.
                if thisStart==0.0:
                    lb = 0.0
                else:
                    lb = thisStart+epsilon
                
                calibrationValue = model.addVar(name="calibrationValue", vtype="C", lb=thisStart+epsilon, ub=thisEnd-epsilon)
                
                varsCorrectClasses = []
                for i, line in enumerate(patternData):
                    for c in range(nofClasses):

                        weightedSum = line[c] - line[c]*weight + line[c+nofClasses]*weight
                        z[i, c] = model.addVar(name=f"z_{i}_{c}", vtype="B")


                        # weightedSum>=calibrationValue -> Z[i,c]
                        # <=> WeightedSum - calibrationValue - Z[i,c] <= 0
                
                        model.addCons(
                                weightedSum - calibrationValue - z[i, c] <= 0,
                                name=f"conformanceSetElementsThere_{i}_{c}",
                            )

                        # For the correct classes also!
                        # weightedSum<calibrationValue -> !Z[i,c]
                        # <=> weightedSum - calibrationValue + (1-Z[i,c]) >= 0
                        # <=> weightedSum - calirationValue - Z[i,c] >= -1

                        if c==int(line[-1]):
                            model.addCons(
                                    weightedSum - calibrationValue - z[i, c] >= -1,
                                    name=f"enoughCoverage_{i}_{c}",
                                )
                            varsCorrectClasses.append(z[i,c])

                # Constraint: Enough coverage
                model.addCons(
                    pyscipopt.quicksum(varsCorrectClasses) >= numberOfCorrectClassifications,
                    name=f"enoughCoverageOverall"
                )
                
                model.setObjective(
                    pyscipopt.quicksum(z[i, c] for i in range(len(patternData)) for c in range(nofClasses)),
                    sense="minimize",
                )

                # model.writeProblem(filename="optimization_problem.lp", trans=False, genericnames=False)
                model.optimize()
                # print("Optimization first step completed...", model.getStatus(), flush=True)
                if (model.getStatus()!="infeasible"):
                    # print("LAST STATUS: ",model.getStatus())
                    solution = model.getBestSol()
                    size = model.getSolObjVal(solution)
                    calibrationValueValue = model.getSolVal(solution,calibrationValue)
                    assert not solution is None
                    z_values = {}
                    for i, line in enumerate(patternData):
                        for c in range(nofClasses):
                            z_values[i,c] = model.getSolVal(solution, z[i,c])

                    # Compure real coverage
                    realCoverage = 0
                    for i, line in enumerate(patternData):
                            realCoverage += z_values[i,int(line[-1])]
    
                    theseRealCoverage.add(realCoverage)
                    
                    model.freeProb()


                    # print("Now find lower/upper bounds of calibration values suitable for this quality")

                    sys.stdout.write("["+str((remainingCalibrationValueIntervals))+"]")
                    sys.stdout.flush()
                    for toMinimize in [True,False]:
                        model = pyscipopt.Model("setextent")
                        model.hideOutput(True)
                        weight = model.addVar(name="weight", vtype="C", lb=0.0, ub=1.0)
                        calibrationValue = model.addVar(name="calibrationValue", vtype="C", lb=thisStart, ub=thisEnd)
                        
                        for i, line in enumerate(patternData):
                            for c in range(nofClasses):

                                weightedSum = line[c] - line[c]*weight + line[c+nofClasses]*weight
                                
                                if c==int(line[-1]) and z_values[i,c]>0.5:
                                    # Enforce acceptance
                                    model.addCons(
                                        weightedSum - calibrationValue >= 0,
                                        name=f"conformanceSetElementsThere_{i}_{c}",
                                    )
                                
                                elif z_values[i,c]<0.5:
                                    # Enforce acceptance
                                    model.addCons(
                                        weightedSum - calibrationValue <= 0 ,
                                        name=f"conformanceSetElementsNotThere_{i}_{c}",
                                    )

                                
                        if toMinimize:
                            model.setObjective(
                                calibrationValue,
                                sense="minimize",
                            )
                            # model.writeProblem(filename="miniWeight.lp", trans=False, genericnames=False)
                            model.optimize()
                            if model.getStatus()=="infeasible":
                                minWeight = calibrationValueValue
                            else:
                                solution = model.getBestSol()
                                minWeight = model.getSolObjVal(solution)
                            
                        else:
                            model.setObjective(
                                calibrationValue,
                                sense="maximize",
                            )
                            model.optimize()
                            # model.writeProblem(filename="miniWeight.lp", trans=False, genericnames=False)
                            # sys.exit(0)
                            if model.getStatus()=="infeasible":
                                maxWeight = calibrationValueValue
                            else:
                                solution = model.getBestSol()
                                maxWeight = model.getSolObjVal(solution)
                                
                        model.freeProb()

                    print("New interval: ",minWeight,maxWeight,size)

                    rest = []
                    if (maxWeight>minWeight): # Only add positive volume
                        thisQualityIntervals.add((minWeight,maxWeight,size))#,tuple(z_values.items())))

                        if (minWeight>thisStart):
                            if (minWeight-thisStart>=epsilon):
                                rest.append((thisStart,minWeight))
                        if maxWeight<thisEnd:
                            if thisEnd-maxWeight>=epsilon:
                                rest.append((maxWeight,thisEnd))
                    else:
                        # Make Intervals smaller
                        if (minWeight>thisStart):
                            if thisStart<minWeight-epsilon:
                                rest.append((thisStart,minWeight-epsilon))
                        if maxWeight<thisEnd:
                            if maxWeight+epsilon<thisEnd:
                                rest.append((maxWeight+epsilon,thisEnd))
                        

                    # print("Previous minimax: ",thisStart,thisEnd)
                    remainingCalibrationValueIntervals = rest + remainingCalibrationValueIntervals[1:]

                    # print("After processing elements with size: ",size,"and interval",minWeight,"to",maxWeight)
                    # print("New remaining calibration value intervals: ",remainingCalibrationValueIntervals)
                else:
                    # Was infeasible
                    model.freeProb()
                    remainingCalibrationValueIntervals = remainingCalibrationValueIntervals[1:]
                        
            allQualityIntervals.append(list(thisQualityIntervals))
            allQualityIntervals[-1].sort()

            if True:
                print("Done with numberOfCorrectClassifications: ",numberOfCorrectClassifications)
                print("Found intervals: ")
                for (a,b,c) in allQualityIntervals[-1]:
                    print("- (",a,"-",b,"):",c)


            if len(theseRealCoverage)>0:
                lastRealCoverage = min(theseRealCoverage)
            else:
                lastRealCoverage = None # Special case: Coverage cannot be achieved at all, possible due to numerics

                        
    return allQualityIntervals



def performCompleteCalibration(inputFile,targetFile):

    # Load input file<
    patterns = []
    thisPatternData = []
    nofLinesLeft = 0
    for line in open(inputFile).readlines():
        line = line.strip()
        if line.startswith("##PATTERN DATA "):
            parts = line.split(" ")
            patternNo = int(parts[2])
            if patternNo != len(patterns):
                print("WARNING: PATTERN NUMBERS DO NOT MATCH.")
            nofLinesLeft = int(parts[3])
        else:
            if nofLinesLeft>0:
                data = [float(a) for a in line.split(" ")]
                thisPatternData.append(data)
                nofLinesLeft-=1
                if nofLinesLeft==0:
                    patterns.append(thisPatternData)
                    thisPatternData = []
        

    # Debuggin
    print("# Patterns:",len(patterns))

    # Process each pattern
    allPatternsQualityData = []
    for patternNo in range(len(patterns)):
        print("Processing pattern no.",patternNo,"with # element:",len(patterns[patternNo]))
        if len(patterns[patternNo])<10:
            allPatternsQualityData.append(processPattern(patterns[patternNo]))
        else:
            allPatternsQualityData.append(processPatternSimplified(patterns[patternNo]))


    with open(targetFile,"w") as outFile:
        outFile.write("#Patterns: "+str(len(allPatternsQualityData)))
        outFile.write("\n")
        for a in allPatternsQualityData:
            outFile.write("#: "+str(len(a)))
            outFile.write("\n")
            for b in a:
                outFile.write("##: "+str(len(b)))
                outFile.write("\n")
                for c in b:
                    outFile.write(str(c[0])+" "+str(c[1])+" "+str(c[2]))
                    outFile.write("\n")


    if False:
        # ================================================  
        # Now compute calibration values that make sense to
        # try. These are the middle points between all
        # points that occur at interval boundaries
        # ================================================
        possibleCalibrationValues = set([])
        for a in allPatternsQualityData: # Patterns
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

        # ===============================================================================
        # Iterate over the calibration values and try to minimize the sum of elements
        # in the conformance sets such that at least "coverageNeeded" many
        # correct ones are covered.
        # ===============================================================================
        for calibrationValue in calibrationValuesToTry:
            print("Trying to compute good solution for calibration value: ",calibrationValue)    
            achievableSizesPerPattern = []
            foundOnlyNonePattern = False # Then due to numerics, we don'T have an estimate for some patterns
            for patternNo in range(len(allPatternsQualityData)):
                achievableSizesPerSet = []
                foundNotNone = False
                for nofCorrectGuesses in range(len(allPatternsQualityData[patternNo])):
                    minNum = None
                    for interval in allPatternsQualityData[patternNo][nofCorrectGuesses]:
                        if (interval[0]<=calibrationValue) and (interval[1]>=calibrationValue):
                            if minNum is None:
                                minNum = interval[2]
                            else:
                                minNum = min(minNum,interval[2])
                    achievableSizesPerSet.append(minNum)
                print("Pattern",patternNo,": achievable set sizes for each coverage number:",achievableSizesPerSet)
                achievableSizesPerPattern.append(achievableSizesPerSet)
                foundOnlyNonePattern |= achievableSizesPerSet[0] is None   # Then, numerics problem

            # Run optimization
            if not foundOnlyNonePattern:
                # # 1. Check how many Pareto frontiers are convex
                convexCases = 0
                nonConvexCases = 0
                for frontier in achievableSizesPerPattern:
                    isConvex = True
                    
                    assert not frontier[0] is None
                    endOfValues = False
                    lastFactor = None
                    for i in range(1, len(frontier)):
                        if endOfValues:
                            if not frontier[i] is None:
                                isConvex = False # Non-monotone due to numerics
                        else:
                            if frontier[i] is None:
                                endOfValues = True
                            else:
                                factor = (frontier[i]-frontier[0])/i
                                if not lastFactor is None:
                                    if factor<lastFactor:
                                        isConvex = False
                                    lastFactor = factor


                    if isConvex:
                        convexCases += 1
                    else:
                        nonConvexCases += 1
                print("#Convex:",convexCases,"of",(convexCases+nonConvexCases))








# ======================= Run =========================
# performCompleteCalibration("outShort.txt","out-short-per-class.txt")
# performCompleteCalibration("testing.txt","testing-achievable-tradeoffs-per-class.txt")
performCompleteCalibration("out-reduced.txt","out-achievable-tradeoffs-per-class.txt")
# performCompleteCalibration("out-only-451.txt","out-451-achievable-tradeoffs-per-class.txt")
# performCompleteCalibration("out-only-460.txt","out-only-460-achievable-tradeoffs-per-class.txt")
# performCompleteCalibration("debug272.txt",4)
