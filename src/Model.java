import Functions.CostFunction;
import Functions.Function;
import Layers.FCLayer;
import Layers.InputLayer;
import Layers.Layer;
import Matrix.Matrix;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Model implements Serializable {
    private List<Layer> layers;
    private int batchSize;
    private CostFunction costFunction;
    private double learningRate;
    private Random rand = new Random(1);

    public Model(int batchSize, CostFunction costFunction, double learningRate) {
        this.layers = new ArrayList<>();
        this.batchSize = batchSize;
        this.costFunction = costFunction;
        this.learningRate = learningRate;
    }

    public void add(int layerSize, Function activationFunction) {
        if (this.layers.size() == 0) {
            this.layers.add(new InputLayer(layerSize));
            return;
        }
        this.layers.add(new FCLayer(this.layers.get(this.layers.size() - 1).getLayerSize(), layerSize,
                activationFunction, this.layers.get(this.layers.size() - 1)));
    }

    public Matrix feedForward(Matrix inputs) {
        Matrix result = inputs.copy();
        for (int i = 0; i < this.layers.size(); i++) {
            result = this.layers.get(i).feedForward(result);
        }
        return result;
    }

    public double backPropagateSample(Matrix inputs, Matrix realValue) {
        Matrix result = this.feedForward(inputs.copy());
        this.layers.get(this.layers.size() - 1).derivativeLayer(this.costFunction.derCost(result, realValue.copy()));
        return this.costFunction.cost(result.copy(), realValue.copy());
    }
    double totalError = 0;
    public void backPropagateBatch(List<Matrix> predictedBatch, List<Matrix> realValueBatch, List<Matrix> validationX,
                                   List<Matrix> validationY) {
        totalError = 0;
        for (Layer layer: this.layers) {
            layer.initGradients();
        }
        for (int i = 0; i < this.batchSize; i++) {
            backPropagateSample(predictedBatch.get(i), realValueBatch.get(i));
        }
        for (Layer layer: this.layers) {
            layer.updateParameters(this.learningRate, this.batchSize);
        }
    }

    public void saveModel(String filename) throws IOException {
        FileOutputStream file = new FileOutputStream(filename);
        ObjectOutputStream out = new ObjectOutputStream(file);

        out.writeObject(this);

        out.close();
        file.close();
        System.out.println("Model saved in " + filename);
    }

    public void setLearningRate(double lr) {
        this.learningRate = lr;
    }

    public static Model loadModel(String filename) throws IOException, ClassNotFoundException {
        FileInputStream file = new FileInputStream(filename);
        ObjectInputStream in = new ObjectInputStream(file);

        // Method for deserialization of object
        Model model = (Model) in.readObject();

        in.close();
        file.close();
        return model;
    }

    public void trainModel(List<Matrix> inputs, List<Matrix> realValues, int epochs, List<Matrix> validationX,
                           List<Matrix> validationY, int printErrorEvery) {
        List<Matrix> shuffledBatches = new ArrayList<>(inputs);
        int nBatches = (int) Math.ceil(inputs.size() / this.batchSize);
        for (int j = 0; j < epochs; j++) {
            Collections.shuffle(shuffledBatches);

            for (int batch = 0; batch < nBatches; batch++) {
                List<Matrix> batchInputs = inputs.subList(batch*this.batchSize, Math.max((batch + 1)*this.batchSize,
                        inputs.size()));
                List<Matrix> batchRealValues = realValues.subList(batch*this.batchSize, Math.max((batch + 1)*this.batchSize,
                        realValues.size()));


                this.backPropagateBatch(batchInputs, batchRealValues, validationX, validationY);
            }


            System.out.print("epoch: " + j + " ");
            boolean printError = (j + 1) % printErrorEvery == 0;
            if (!printError) {
                System.out.println("");
                return;
            }
            for (int i = 0; i < validationX.size(); i++) {
                totalError += this.costFunction.cost(this.feedForward(validationX.get(i)), validationY.get(i));
            }
            totalError /= validationX.size();
            System.out.println(" Total error = " + totalError);

            // to plot the regression demo
//            RegressionEx.writePointFile(this, j);

            // to plot the isInCircle demo
            IsInCircle.writePointFile(this, j);


//            int testSize = validationX.size();
//            List<Matrix> XTest = validationX;
//            Double[] xPredictsTrue = new Double[testSize];
//            Double[] yPredictsTrue = new Double[testSize];
//            Double[] xPredictsFalse = new Double[testSize];
//            Double[] yPredictsFalse = new Double[testSize];
//            for (int i = 0; i < testSize; i++) {
//                if (this.feedForward(XTest.get(i)).getValues()[0][0] > 0) {
//                    xPredictsTrue[i] = XTest.get(i).getValues()[0][0];
//                    yPredictsTrue[i] = XTest.get(i).getValues()[1][0];
//                } else {
//                    xPredictsFalse[i] = XTest.get(i).getValues()[0][0];
//                    yPredictsFalse[i] = XTest.get(i).getValues()[1][0];
//                }
//            }
//            PointsWriter pointsWriter = new PointsWriter(xPredictsTrue, yPredictsTrue, testSize, "predicted-true" + j + ".txt");
//            pointsWriter.writePoints();
//            pointsWriter = new PointsWriter(xPredictsFalse, yPredictsFalse, testSize, "predicted-false" + j + ".txt");
//            pointsWriter.writePoints();

        }
    }
}
