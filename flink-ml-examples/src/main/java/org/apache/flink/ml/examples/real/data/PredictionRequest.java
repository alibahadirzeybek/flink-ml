package org.apache.flink.ml.examples.real.data;

import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.annotation.JsonProperty;

/** HH. */
public class PredictionRequest {
    @JsonProperty("search_loc")
    public String searchLocation;

    public PredictionRequest(@JsonProperty("search_loc") String searchLocation) {
        this.searchLocation = searchLocation;
    }
}
