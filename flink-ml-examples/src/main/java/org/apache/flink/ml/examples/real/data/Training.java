package org.apache.flink.ml.examples.real.data;

import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.annotation.JsonProperty;

/** HH. */
public class Training {
    @JsonProperty("search_loc")
    public String searchLocation;

    @JsonProperty("chip_name")
    public String chipName;

    public Training(
            @JsonProperty("search_loc") String searchLocation,
            @JsonProperty("chip_name") String chipName) {
        this.searchLocation = searchLocation;
        this.chipName = chipName;
    }
}
