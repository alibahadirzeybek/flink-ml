/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.examples.classification;

import org.apache.flink.ml.classification.logisticregression.LogisticRegression;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains an OnlineLogisticRegression model and uses it for classification. */
public class LogisticRegressionExampleV2 {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(Vectors.dense(1.0), 2.0),
                        Row.of(Vectors.dense(1.0), 2.0),
                        Row.of(Vectors.dense(1.0), 2.0),
                        Row.of(Vectors.dense(1.0), 3.0),
                        Row.of(Vectors.dense(1.0), 3.0),
                        Row.of(Vectors.dense(1.0), 3.0),
                        Row.of(Vectors.dense(1.0), 4.0),
                        Row.of(Vectors.dense(1.0), 4.0),
                        Row.of(Vectors.dense(1.0), 4.0),
                        Row.of(Vectors.dense(1.0), 5.0));
        Table inputTable = tEnv.fromDataStream(inputStream).as("features", "label");

        // Creates a LogisticRegression object and initializes its parameters.
        LogisticRegression lr = new LogisticRegression();

        // Trains the LogisticRegression Model.
        LogisticRegressionModel lrModel = lr.fit(inputTable);

        // Uses the LogisticRegression Model for predictions.
        Table outputTable = lrModel.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector features = (DenseVector) row.getField(lr.getFeaturesCol());
            double expectedResult = (Double) row.getField(lr.getLabelCol());
            double predictionResult = (Double) row.getField(lr.getPredictionCol());
            DenseVector rawPredictionResult = (DenseVector) row.getField(lr.getRawPredictionCol());
            System.out.printf(
                    "Features: %-25s \tExpected Result: %s \tPrediction Result: %s \tRaw Prediction Result: %s\n",
                    features, expectedResult, predictionResult, rawPredictionResult);
        }
    }
}
