package org.apache.flink.ml.examples.real.data;

import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.annotation.JsonProperty;

/** HH. */
public class PredictionResponse {
    @JsonProperty("search_loc")
    public String searchLocation;

    @JsonProperty("user_preferred_chips")
    public String[] userPreferredChips;

    public PredictionResponse(
            @JsonProperty("search_loc") String searchLocation,
            @JsonProperty("user_preferred_chips") String[] userPreferredChips) {
        this.searchLocation = searchLocation;
        this.userPreferredChips = userPreferredChips;
    }
}
