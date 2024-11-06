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

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.classification.logisticregression.LogisticRegression;
import org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegression;
import org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegressionModel;
import org.apache.flink.ml.examples.util.BoundedPeriodicSourceFunction;
import org.apache.flink.ml.examples.util.PeriodicSourceFunction;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** Simple program that trains an OnlineLogisticRegression model and uses it for classification. */
public class OnlineLogisticRegressionExampleV3 {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data. Both are infinite streams that periodically
        // sends out provided data to trigger model update and prediction.
        List<Row> trainDataInitial =
                Arrays.asList(
                        Row.of(Vectors.dense(1.0), 1.0),
                        Row.of(Vectors.dense(1.0), 1.0),
                        Row.of(Vectors.dense(1.0), 1.0),
                        Row.of(Vectors.dense(1.0), 1.0),
                        Row.of(Vectors.dense(1.0), 1.0),
                        Row.of(Vectors.dense(1.0), 1.0),
                        Row.of(Vectors.dense(1.0), 1.0),
                        Row.of(Vectors.dense(1.0), 1.0),
                        Row.of(Vectors.dense(1.0), 1.0),
                        Row.of(Vectors.dense(1.0), 1.0));

        List<Row> trainDataIncremental =
                Arrays.asList(
                        Row.of(Vectors.dense(1.0), 0.0),
                        Row.of(Vectors.dense(1.0), 0.0),
                        Row.of(Vectors.dense(1.0), 0.0),
                        Row.of(Vectors.dense(1.0), 0.0),
                        Row.of(Vectors.dense(1.0), 0.0),
                        Row.of(Vectors.dense(1.0), 0.0),
                        Row.of(Vectors.dense(1.0), 0.0),
                        Row.of(Vectors.dense(1.0), 0.0),
                        Row.of(Vectors.dense(1.0), 0.0),
                        Row.of(Vectors.dense(1.0), 0.0));

        List<Row> predictData =
                Collections.singletonList(Row.of(Vectors.dense(1.0), 0.0));

        RowTypeInfo typeInfo =
                new RowTypeInfo(
                        new TypeInformation[] {DenseVectorTypeInfo.INSTANCE, Types.DOUBLE},
                        new String[] {"features", "label"});

        SourceFunction<Row> trainSource =
                new BoundedPeriodicSourceFunction(
                        1_000L,
                        Arrays.asList(
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataIncremental,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial,
                                trainDataInitial));
        DataStream<Row> trainStream = env.addSource(trainSource, typeInfo);
        Table trainTable = tEnv.fromDataStream(trainStream).as("features");

        SourceFunction<Row> predictSource =
                new PeriodicSourceFunction(100L, Collections.singletonList(predictData));
        DataStream<Row> predictStream = env.addSource(predictSource, typeInfo);
        Table predictTable = tEnv.fromDataStream(predictStream).as("features");

        // Creates an online LogisticRegression object and initializes its parameters and initial
        // model data.
        Table initModelDataTable =
                new LogisticRegression()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(10)
                        .fit(tEnv.fromDataStream(
                                env.fromCollection(trainDataInitial, typeInfo)
                            )
                        )
                        .getModelData()[0];
        OnlineLogisticRegression olr =
                new OnlineLogisticRegression()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(10)
                        .setInitialModelData(initModelDataTable);

        // Trains the online LogisticRegression Model.
        OnlineLogisticRegressionModel onlineModel = olr.fit(trainTable);

        // Uses the online LogisticRegression Model for predictions.
        Table outputTable = onlineModel.transform(predictTable)[0];

        // Extracts and displays the results. As training data stream continuously triggers the
        // update of the internal model data, raw prediction results of the same predict dataset
        // would change over time.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector features = (DenseVector) row.getField(olr.getFeaturesCol());
            Long modelVersion = (Long) row.getField(olr.getModelVersionCol());
            Double expectedResult = (Double) row.getField(olr.getLabelCol());
            Double predictionResult = (Double) row.getField(olr.getPredictionCol());
            DenseVector rawPredictionResult = (DenseVector) row.getField(olr.getRawPredictionCol());
            System.out.printf(
                    "Features: %-25s \tModel Version: %s  \tExpected Result: %s \tPrediction Result: %s \tRaw Prediction Result: %s\n",
                    features, modelVersion, expectedResult, predictionResult, rawPredictionResult);
        }
    }
}
