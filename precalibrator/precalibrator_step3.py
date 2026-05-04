#!/usr/bin/env python3
import argparse
import sys, math, copy
import pyscipopt 


DEFAULT_SCIP_TIME_LIMIT = 1200.0 # 20 Min
DEFAULT_SCIP_MIP_GAP = 1e-3


def _configure_scip_model(model, scip_time_limit, scip_mip_gap):
    if scip_time_limit is not None:
        model.setParam("limits/time", float(scip_time_limit))
    if scip_mip_gap is not None:
        model.setParam("limits/gap", float(scip_mip_gap))


def processPattern(patternData,calibrationValue,requirementsOnCoveredCases,):
    """Process individual pattern."""

    epsilon = 0.00001
    allQualityIntervals = []
    lastRealCoverage = None
    nofClasses = (len(patternData[0])-1)//2
    
    # Build MILP instance
    model = pyscipopt.Model("findset")
    model.hideOutput(True)
    _configure_scip_model(model, DEFAULT_SCIP_TIME_LIMIT, DEFAULT_SCIP_MIP_GAP)
    z = {}
    z_values = {}
    weight = model.addVar(name="weight", vtype="C", lb=0.0, ub=1.0)
    
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
        pyscipopt.quicksum(varsCorrectClasses) >= requirementsOnCoveredCases,
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
        if solution is None:
            raise Exception(
                "No incumbent solution found in precalibrator step 3. "
                f"SCIP status: {model.getStatus()}."
            )
        weightValue = model.getSolVal(solution,weight)
        sizeConformanceSets = 0
        for i, line in enumerate(patternData):
            for c in range(nofClasses):
                z_values[i,c] = model.getSolVal(solution, z[i,c])
                if z_values[i,c]>0.5:
                    sizeConformanceSets += 1

        # Compure real coverage
        realCoverage = 0
        for i, line in enumerate(patternData):
            realCoverage += z_values[i,int(line[-1])]

        print("Weight for pattern: ",weightValue,"with coverage",realCoverage,"and size of conformance sets",sizeConformanceSets)
        return {
            "weight": float(weightValue),
            "coverage": float(realCoverage),
            "size": int(sizeConformanceSets),
        }
    else:
        print("Error: NO SOLUTION FOUND.")
        raise Exception("No Solution Found")



def performCalibrationStep3(inputFile,step2OutputFile,outputFile=None,):

    # Load input file
    patterns = []
    thisPatternData = []
    nofLinesLeft = 0
    for line in open(inputFile).readlines():
        line = line.strip()
        if line.startswith("Case: "):
            continue
        if line.startswith("##PATTERN DATA "):
            parts = line.split(" ")
            patternNo = int(parts[2])
            assert patternNo == len(patterns)
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

    with open(step2OutputFile,"r") as inFile:
        calibrationValue = float(inFile.readline().strip())
        requirementsOnCoveredCases = []
        for a in inFile.readlines():
            a = a.strip()
            requirementsOnCoveredCases.append(int(a))

    if len(requirementsOnCoveredCases) != len(patterns):
        raise ValueError(
            "Mismatch between number of exported patterns and step-2 coverage requirements."
        )

    print(
        f"[step3] SCIP limits: time={DEFAULT_SCIP_TIME_LIMIT}s, mip_gap={DEFAULT_SCIP_MIP_GAP}",
        flush=True,
    )
        
    results = []
    for i,a in enumerate(requirementsOnCoveredCases):
        print("Processing pattern:",i,"with number of data points",len(patterns[i]))
        results.append(processPattern(patterns[i],calibrationValue,a,))

    if outputFile is not None:
        with open(outputFile, "w", encoding="utf-8") as outFile:
            outFile.write(f"calibration_value {calibrationValue}\n")
            for i, result in enumerate(results):
                outFile.write(
                    f"{i} {result['weight']} {result['coverage']} {result['size']}\n"
                )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="?", default="outShort.txt")
    parser.add_argument(
        "step2_output_file",
        nargs="?",
        default="out-short-per-class-input-to-step-3.txt",
    )
    parser.add_argument("output_file", nargs="?", default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    performCalibrationStep3(
        args.input_file,
        args.step2_output_file,
        args.output_file,
    )
