package org.apache.flink.ml.examples.real.util;

import org.apache.flink.ml.examples.real.data.Training;
import org.apache.flink.types.Row;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** HH. */
public class Generator {
    public static List<Row> getInitialModelData() {
        return Stream.of(
                        new Training("New Orleans, LA", "Luxury"),
                        new Training("New Orleans, LA", "Comfort"),
                        new Training("Las Vegas, NV", "Luxury"),
                        new Training("Las Vegas, NV", "Comfort"),
                        new Training("New York, NY", "Luxury"),
                        new Training("New York, NY", "Comfort"))
                .map(
                        training ->
                                Row.of(
                                        Encodings.getFeatureVectorEncoding(training.searchLocation),
                                        Encodings.getLabelEncoding(training.chipName)))
                .collect(Collectors.toList());
    }
}
