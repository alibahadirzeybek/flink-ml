package org.apache.flink.ml.examples.real.util;

import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;

/** HH. */
public class Encodings {
    public static DenseVector getFeatureVectorEncoding(String featureName) {
        switch (featureName) {
            case "New Orleans, LA":
                return Vectors.dense(1.0, 0.0, 0.0);
            case "Las Vegas, NV":
                return Vectors.dense(0.0, 1.0, 0.0);
            case "New York, NY":
                return Vectors.dense(0.0, 0.0, 1.0);
            default:
                throw new RuntimeException(String.join(": ", "Unknown feature name", featureName));
        }
    }

    public static String getFeatureNameEncoding(DenseVector featureVector) {
        if (featureVector.get(0) == 1.0) {
            return "New Orleans, LA";
        } else if (featureVector.get(1) == 1.0) {
            return "Las Vegas, NV";
        } else if (featureVector.get(2) == 1.0) {
            return "New York, NY";
        } else {
            throw new RuntimeException(
                    String.join(": ", "Unknown feature vector", featureVector.toString()));
        }
    }

    public static Double getLabelEncoding(String labelName) {
        switch (labelName) {
            case "Luxury":
                return 0.0;
            case "Comfort":
                return 1.0;
            default:
                throw new RuntimeException(String.join(": ", "Unknown label name", labelName));
        }
    }

    public static String[] getLabelNameEncoding(DenseVector labelRawPredictions) {
        if (labelRawPredictions.get(0) > labelRawPredictions.get(1)) {
            return new String[] {"Luxury", "Comfort"};
        } else {
            return new String[] {"Comfort", "Luxury"};
        }
    }
}
