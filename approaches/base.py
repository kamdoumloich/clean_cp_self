#
# Conformance predictor implementation.
# Note that all conformance predictors need to return 0 classes with calibration value 0, all classes with calibration value 1, and be monotone in between.
# They return classes and from which calibration value the class set would increase in size (None if the set has all classes).
#
import math

import numpy as np

class MonotoneConformanceEvaluatorSingleDistributionAllAboveThreshold:
    def __init__(self):
        pass

    def predict(self,probabilityDistributions,calibrationValue):
        """Standard conformance predictor. Returns all classes whose probabilities are above 1 Minus the calibration value."""
        resultSet = []
        nextOne = None
        for i in range(len(probabilityDistributions[0])):
            #print("CV:",calibrationValue)
            #print("PD:",probabilityDistributions[0])
            if 1.0-probabilityDistributions[0][i]<=calibrationValue:
                resultSet.append(i)
            else:
                if nextOne is None:
                    nextOne = 1.0-probabilityDistributions[0][i]
                    assert not nextOne==0.0
                else:
                    nextOne = min(nextOne,1.0-probabilityDistributions[0][i])
                    assert not nextOne==0.0
        return resultSet,nextOne

    def texInfo(self):
        return """Basic single-distribution higher-than-threshold conformance predictor."""

    def getShortName(self):
        return "Base-1D"


class MonotoneConformanceEvaluatorSingleDistributionAllAboveThresholdButAtLeastOneClass:
    def __init__(self):
        pass

    def predict(self,probabilityDistributions,calibrationValue):
        """Standard conformance predictor. Returns all classes whose probabilities are above 1 Minus the calibration value."""
        resultSet = []
        nextOne = None
        maxClassValue = 0.0
        maxClass = None
        for i in range(len(probabilityDistributions[0])):
            #print("CV:",calibrationValue)
            #print("PD:",probabilityDistributions[0])
            if probabilityDistributions[0][i]>maxClassValue:
                maxClassValue = probabilityDistributions[0][i]
                maxClass = i
            if 1.0-probabilityDistributions[0][i]<=calibrationValue:
                resultSet.append(i)
            else:
                if nextOne is None:
                    nextOne = 1.0-probabilityDistributions[0][i]
                    assert not nextOne==0.0
                else:
                    nextOne = min(nextOne,1.0-probabilityDistributions[0][i])
                    assert not nextOne==0.0
        if len(resultSet)==0:
            resultSet = [maxClass]
        return resultSet,nextOne

    def texInfo(self):
        return """Basic single-distribution higher-than-threshold conformance predictor, but the returned set needs to have at least one class"""

    def getShortName(self):
        return "Base-1D-AtLeastOneClass"

    def precalibrate(self,):
        pass

    def get_score_fitted_model(self, test_data):
        pass

class APSConformanceEvaluatorSingleDistribution:
    def __init__(self):
        pass

    def predict(self, probabilityDistributions, calibrationValue):

        assert len(probabilityDistributions) >= 2
        resultSet = []
        restClasses = set(range(0, len(probabilityDistributions[0])))
        done = False
        probabilityCoveredSoFar = 0.0
        prob_dist = probabilityDistributions[0].copy()

        while (probabilityCoveredSoFar <= calibrationValue) and len(restClasses) > 0:
            # Search for the remaining class Combination with the highest value

            if max(prob_dist) <= 0:
                done = True
                break
            probabilityCoveredSoFar += max(prob_dist)
            resultSet.extend([int(np.argmax(prob_dist))])
            prob_dist[np.argmax(prob_dist)] = -10
            restClasses.difference_update(resultSet)

        # # Numerics Fall-Back: If probability sum was already 1 or above, add the rest of the classes
        if ((calibrationValue == 1.0) or done) and len(restClasses) > 0:
            resultSet.extend(restClasses)

        return resultSet, math.nextafter(probabilityCoveredSoFar, math.inf)

    def texInfo(self):
        return """Basic single-distribution higher-than-threshold conformance predictor, but the returned set needs to have at least one class"""

    def getShortName(self):
        return "Base-1D-AtLeastOneClass"

    def precalibrate(self, precalibration_data):
        pass

    def get_score_fitted_model(self, test_data):
        pass
