package org.apache.flink.ml.examples.real;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.connector.base.DeliveryGuarantee;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.formats.json.JsonDeserializationSchema;
import org.apache.flink.formats.json.JsonSerializationSchema;
import org.apache.flink.ml.classification.logisticregression.LogisticRegression;
import org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegression;
import org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegressionModel;
import org.apache.flink.ml.examples.real.data.PredictionRequest;
import org.apache.flink.ml.examples.real.data.PredictionResponse;
import org.apache.flink.ml.examples.real.data.Training;
import org.apache.flink.ml.examples.real.util.Encodings;
import org.apache.flink.ml.examples.real.util.Generator;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.PrintSink;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import java.util.Objects;

/** Simple program that trains an OnlineLogisticRegression model and uses it for classification. */
public class DynamicChipElements {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment streamExecutionEnvironment =
                StreamExecutionEnvironment.getExecutionEnvironment();

        streamExecutionEnvironment.setParallelism(1);

        StreamTableEnvironment streamTableEnvironment =
                StreamTableEnvironment.create(streamExecutionEnvironment);

        RowTypeInfo typeInfoTraining =
                new RowTypeInfo(
                        new TypeInformation[] {DenseVectorTypeInfo.INSTANCE, Types.DOUBLE},
                        new String[] {"features", "label"});

        RowTypeInfo typeInfoPredictionRequest =
                new RowTypeInfo(
                        new TypeInformation[] {DenseVectorTypeInfo.INSTANCE},
                        new String[] {"features"});

        KafkaSource<Training> trainingKafkaSource =
                KafkaSource.<Training>builder()
                        .setBootstrapServers("FILL_ME_HERE")
                        .setTopics("search_two_chip_train")
                        .setGroupId("ververica")
                        .setStartingOffsets(OffsetsInitializer.earliest())
                        .setValueOnlyDeserializer(new JsonDeserializationSchema<>(Training.class))
                        .setProperty("security.protocol", "SSL")
                        .setProperty("ssl.truststore.type", "JKS")
                        .setProperty("ssl.truststore.location", "FILL_ME_HERE")
                        .setProperty("ssl.truststore.password", "FILL_ME_HERE")
                        .setProperty("ssl.keystore.type", "PKCS12")
                        .setProperty("ssl.keystore.location", "FILL_ME_HERE")
                        .setProperty("ssl.keystore.password", "FILL_ME_HERE")
                        .build();

        KafkaSource<PredictionRequest> predictionRequestKafkaSource =
                KafkaSource.<PredictionRequest>builder()
                        .setBootstrapServers("FILL_ME_HERE")
                        .setTopics("search_two_chip_predict_req")
                        .setGroupId("ververica")
                        .setStartingOffsets(OffsetsInitializer.latest())
                        .setValueOnlyDeserializer(
                                new JsonDeserializationSchema<>(PredictionRequest.class))
                        .setProperty("security.protocol", "SSL")
                        .setProperty("ssl.truststore.type", "JKS")
                        .setProperty("ssl.truststore.location", "FILL_ME_HERE")
                        .setProperty("ssl.truststore.password", "FILL_ME_HERE")
                        .setProperty("ssl.keystore.type", "PKCS12")
                        .setProperty("ssl.keystore.location", "FILL_ME_HERE")
                        .setProperty("ssl.keystore.password", "FILL_ME_HERE")
                        .build();

        DataStream<Row> trainingDataStream =
                streamExecutionEnvironment
                        .fromSource(
                                trainingKafkaSource,
                                WatermarkStrategy.noWatermarks(),
                                "training-kafka-source")
                        .map(
                                training ->
                                        Row.of(
                                                Encodings.getFeatureVectorEncoding(
                                                        training.searchLocation),
                                                Encodings.getLabelEncoding(training.chipName)),
                                typeInfoTraining);

        DataStream<Row> predictionRequestDataStream =
                streamExecutionEnvironment
                        .fromSource(
                                predictionRequestKafkaSource,
                                WatermarkStrategy.noWatermarks(),
                                "prediction-request-kafka-source")
                        .map(
                                predictionRequest ->
                                        Row.of(
                                                Encodings.getFeatureVectorEncoding(
                                                        predictionRequest.searchLocation)),
                                typeInfoPredictionRequest);

        Table trainingTable = streamTableEnvironment.fromDataStream(trainingDataStream);

        Table predictionRequestTable =
                streamTableEnvironment.fromDataStream(predictionRequestDataStream);

        Table initialModelDataTable =
                new LogisticRegression()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(10)
                        .fit(
                                streamTableEnvironment.fromDataStream(
                                        streamExecutionEnvironment.fromCollection(
                                                Generator.getInitialModelData(), typeInfoTraining)))
                        .getModelData()[0];

        OnlineLogisticRegression onlineLogisticRegression =
                new OnlineLogisticRegression()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(10)
                        .setInitialModelData(initialModelDataTable);

        OnlineLogisticRegressionModel onlineLogisticRegressionModel =
                onlineLogisticRegression.fit(trainingTable);

        Table predictionResponseTable =
                onlineLogisticRegressionModel.transform(predictionRequestTable)[0];

        DataStream<Row> predictionResponseDataStream =
                streamTableEnvironment.toDataStream(predictionResponseTable);

        predictionResponseDataStream.sinkTo(new PrintSink<>());

        KafkaSink<PredictionResponse> predictionResponseKafkaSink =
                KafkaSink.<PredictionResponse>builder()
                        .setBootstrapServers("FILL_ME_HERE")
                        .setRecordSerializer(
                                KafkaRecordSerializationSchema.<PredictionResponse>builder()
                                        .setTopic("search_two_chip_predict_res")
                                        .setValueSerializationSchema(
                                                new JsonSerializationSchema<PredictionResponse>())
                                        .build())
                        .setDeliveryGuarantee(DeliveryGuarantee.AT_LEAST_ONCE)
                        .setProperty("security.protocol", "SSL")
                        .setProperty("ssl.truststore.type", "JKS")
                        .setProperty("ssl.truststore.location", "FILL_ME_HERE")
                        .setProperty("ssl.truststore.password", "FILL_ME_HERE")
                        .setProperty("ssl.keystore.type", "PKCS12")
                        .setProperty("ssl.keystore.location", "FILL_ME_HERE")
                        .setProperty("ssl.keystore.password", "FILL_ME_HERE")
                        .build();

        predictionResponseDataStream
                .map(
                        predictionResponse ->
                                new PredictionResponse(
                                        Encodings.getFeatureNameEncoding(
                                                (DenseVector)
                                                        Objects.requireNonNull(
                                                                predictionResponse.getField(
                                                                        "features"))),
                                        Encodings.getLabelNameEncoding(
                                                (DenseVector)
                                                        Objects.requireNonNull(
                                                                predictionResponse.getField(
                                                                        "rawPrediction")))))
                .sinkTo(predictionResponseKafkaSink);

        streamExecutionEnvironment.execute("dynamic-filter-elements");
    }
}
