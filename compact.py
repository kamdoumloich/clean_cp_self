#!/usr/bin/env python3
import os, math

import numpy as np


def runCompleteEvaluation(texTargetFile,benchmark,conformanceApproaches,interestingCaseFilter):
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
            for x in range(1,len(benchmark.classes)):
                outFile.write("\\draw[color=black!50!white,dashed] (0,"+str(conformanceAlgoNo*10+5+5.0/len(benchmark.classes)*x)+") -- +(10,0);\n")
                outFile.write("\\draw[color=black!50!white] (10,"+str(conformanceAlgoNo*10+5+5.0/len(benchmark.classes)*x)+") -- +(0.2,0) node[right] {\\scriptsize "+str(x)+"};\n")


        # 3. Build the accuracy/set size trade-off curves
        for conformanceAlgoNo in range(0,len(conformanceApproaches)):

            ################ PERFORM PRECALIBRATION ######################
            # algoOject = conformanceApproaches[conformanceAlgoNo]
            print("Case: ", conformanceApproaches[conformanceAlgoNo].getShortName())
            conformanceApproaches[conformanceAlgoNo].precalibrate()

            currentParameter = 0.0
            data = []
            done = False
            while not done:
                correctCases = 0 
                sumSizes = 0
                nextParameter = 1.0
                allCases = 0
                interestingCaseFilter.reportNewContext(conformanceAlgoNo,currentParameter)
                for i,(probabilities,classification) in enumerate(benchmark.calibration_data):
                    classes,nextUp = conformanceApproaches[conformanceAlgoNo].predict(probabilities,currentParameter)
                        # state = ""
                    sumSizes += len(classes)
                    allCases += 1
                    if classification in classes:
                        # Correct!
                        correctCases +=1
                    else:
                        nextParameter = min(nextUp,nextParameter)
                    interestingCaseFilter.reportCase(i,probabilities,classification,classes)
                percentage = correctCases/allCases
                # print("Current Param",currentParameter," correct: ",correctCases,allCases)
                # print("Next parameter: ",nextParameter)
                interestingCaseFilter.reportProbabilityOfBeingCorrect(percentage)
                # Compute Alpha from the percentage
                realLevel = percentage*allCases/(allCases+1)
                data.append((currentParameter,realLevel,sumSizes/allCases))

                if correctCases==allCases:
                    done = True
                else:
                    assert currentParameter!=nextParameter # Finds bugs in the approaches
                    currentParameter=nextParameter
            
            # Draw corresponding curves
            # a) Real probability of the conformance predictor being correct
            outFile.write("\\draw[color=red,thick] (0,"+str(conformanceAlgoNo*10)+")")
            for i,a in enumerate(data):
                (parameter,level,meanSize) = a
                outFile.write(" |- ");
                outFile.write("("+str(getPercentageCoord(level))+","+str(getPercentageCoordY(level)+conformanceAlgoNo*10)+")")
            outFile.write(";\n")
            # b) Mean sizes of the conformance sets
            outFile.write("\\draw[color=blue,thick] (0,"+str(conformanceAlgoNo*10+5)+")")
            for i,a in enumerate(data):
                (parameter,level,meanSize) = a
                outFile.write(" |- ");
                outFile.write("("+str(getPercentageCoord(level))+","+str(meanSize*5/len(benchmark.classes)+5+conformanceAlgoNo*10)+")")
            outFile.write(";\n")

        # X. End Comparison
        outFile.write("\\end{tikzpicture}")

        # X+1 Interesting cases found
        outFile.write("\\section{Interesting cases found Calibration}\n")
        outFile.write(interestingCaseFilter.getTexReport());

        # Testing
        outFile.write("\\section{Performance on test data}\n")

        listCalibration = []
        for conformanceAlgoNo in range(0, len(conformanceApproaches)):
            listCalibration.append(interestingCaseFilter.getCalibrationParameter(predictorNumber=conformanceAlgoNo))

        K = len(benchmark.classes)
        interestingCaseFilter.resetFilter()

        # IMPORTANT: ensure calibrationParameter is fetched before reset, unless resetFilter preserves it.
        # calibrationParameter = interestingCaseFilter.getCalibrationParameter(...)

        for conformanceAlgoNo in range(len(conformanceApproaches)):
            correctCases = 0
            sumSizes = 0
            allCases = 0
            calibrationParameter = listCalibration[conformanceAlgoNo]

            n_set_of_size = np.zeros(K + 1, dtype=int)  # 0..K

            interestingCaseFilter.reportNewContext(conformanceAlgoNo, calibrationParameter)

            for i, (probabilities, classification) in enumerate(benchmark.testing_data):
                classes, _ = conformanceApproaches[conformanceAlgoNo].predict(probabilities, calibrationParameter)
                s = len(classes)

                sumSizes += s
                allCases += 1

                if s <= K:
                    n_set_of_size[s] += 1
                else:
                    # should never happen; indicates a bug in predict()
                    raise ValueError(f"Predicted set size {s} > number of classes {K}")

                if classification in classes:
                    correctCases += 1

                interestingCaseFilter.reportCase(i, probabilities, classification, classes)

            if allCases == 0:
                outFile.write(f"\\paragraph{{Predictor:{conformanceAlgoNo}}} No test cases.\\\\\n")
                continue

            coverage = correctCases / allCases
            avg_size = sumSizes / allCases

            outFile.write(
                f"\\paragraph{{Predictor:{conformanceAlgoNo}}} Calibration value: {calibrationParameter} "
                f"Correct: {correctCases}/{allCases}\\\\\n"
            )
            outFile.write(f"Average set size: {avg_size}   Coverage: {coverage}\\\\\n")
            outFile.write("Set size histogram (0..K):\\\\\n")
            outFile.write(" | ".join(map(str, n_set_of_size.tolist())))
            outFile.write("\\\\\n")

            interestingCaseFilter.reportProbabilityOfBeingCorrect(coverage)

        # X+1 Interesting cases found
        outFile.write("\\section{Interesting cases found Testing}\n")
        outFile.write(interestingCaseFilter.getTexReport());

        # End
        outFile.write("""
        \\end{document}
        """)
