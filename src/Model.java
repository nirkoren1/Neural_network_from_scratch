import Functions.CostFunction;
import Functions.Function;
import Layers.FCLayer;
import Layers.InputLayer;
import Layers.Layer;
import Matrix.Matrix;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Model {
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

    public double backPorpegateSample(Matrix inputs, Matrix realValue) {
        Matrix result = this.feedForward(inputs.copy());
        this.layers.get(this.layers.size() - 1).derivativeLayer(this.costFunction.derCost(result, realValue.copy()));
        return this.costFunction.cost(result.copy(), realValue.copy());
    }
    double totalError = 0;
    public void backPorpegateBatch(List<Matrix> predictedBatch, List<Matrix> realValueBatch, List<Matrix> validationX,
                                   List<Matrix> validationY) {
        totalError = 0;
        for (Layer layer: this.layers) {
            layer.initGradients();
        }
        for (int i = 0; i < this.batchSize; i++) {
            backPorpegateSample(predictedBatch.get(i), realValueBatch.get(i));
        }
        for (Layer layer: this.layers) {
            layer.updateParameters(this.learningRate, this.batchSize);
        }
        for (int i = 0; i < validationX.size(); i++) {
            double xPredict = feedForward(validationX.get(i)).getValues()[0][0];
            double yTrue = validationY.get(i).getValues()[0][0];
            totalError += Math.pow(xPredict - yTrue, 2);
        }
        totalError /= validationX.size();
        System.out.println(" Total error = " + totalError);
    }

    public void trainModel(List<Matrix> inputs, List<Matrix> realValues, int epochs, List<Matrix> validationX,
                           List<Matrix> validationY) {
        for (int j = 0; j < epochs; j++) {
            List<Matrix> batchInputs = new ArrayList<>();
            List<Matrix> batchRealValues = new ArrayList<>();
            for (int i = 0; i < this.batchSize; i++) {
                int randIndex = rand.nextInt(inputs.size());
                batchInputs.add(inputs.get(randIndex));
                batchRealValues.add(realValues.get(randIndex));
            }
            System.out.print("epoch: " + j + " ");
            this.backPorpegateBatch(batchInputs, batchRealValues, validationX, validationY);
//            RegressionEx.writePointFile(this, j);
            int testSize = validationX.size();
            List<Matrix> XTest = validationX;
            Double[] xPredictsTrue = new Double[testSize];
            Double[] yPredictsTrue = new Double[testSize];
            Double[] xPredictsFalse = new Double[testSize];
            Double[] yPredictsFalse = new Double[testSize];
            for (int i = 0; i < testSize; i++) {
                if (this.feedForward(XTest.get(i)).getValues()[0][0] > 0) {
                    xPredictsTrue[i] = XTest.get(i).getValues()[0][0];
                    yPredictsTrue[i] = XTest.get(i).getValues()[1][0];
                } else {
                    xPredictsFalse[i] = XTest.get(i).getValues()[0][0];
                    yPredictsFalse[i] = XTest.get(i).getValues()[1][0];
                }
            }
            PointsWriter pointsWriter = new PointsWriter(xPredictsTrue, yPredictsTrue, testSize, "predicted-true" + j + ".txt");
            pointsWriter.writePoints();
            pointsWriter = new PointsWriter(xPredictsFalse, yPredictsFalse, testSize, "predicted-false" + j + ".txt");
            pointsWriter.writePoints();
        }
    }
}
