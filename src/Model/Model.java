package Model;

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
    private Random rand = new Random();

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

    public void backPorpegateSample(Matrix inputs, Matrix realValue) {
        Matrix result = this.feedForward(inputs.copy());
        this.layers.get(this.layers.size() - 1).derivativeLayer(this.costFunction.cost(result, realValue.copy()));
    }

    public void backPorpegateBatch(List<Matrix> predictedBatch, List<Matrix> realValueBatch) {
        for (Layer layer: this.layers) {
            layer.initGradients();
        }
        for (int i = 0; i < this.batchSize; i++) {
            backPorpegateSample(predictedBatch.get(i), realValueBatch.get(i));
        }
        for (Layer layer: this.layers) {
            layer.updateParameters(this.learningRate, this.batchSize);
        }
        double totalError = 0;
        for (int i = 0; i < this.batchSize; i++) {
            totalError += Math.abs(this.feedForward(predictedBatch.get(i)).getValues()[0][0] - realValueBatch.get(i).getValues()[0][0]);
        }
        totalError /= this.batchSize;
        System.out.println(" Total error = " + totalError);
    }

    public void trainModel(List<Matrix> inputs, List<Matrix> realValues, int epochs) {
        for (int j = 0; j < epochs; j++) {
            List<Matrix> batchInputs = new ArrayList<>();
            List<Matrix> batchRealValues = new ArrayList<>();
            for (int i = 0; i < this.batchSize; i++) {
                int randIndex = rand.nextInt(inputs.size());
                batchInputs.add(inputs.get(randIndex));
                batchRealValues.add(realValues.get(randIndex));
            }
            System.out.print("epoch: " + j);
            this.backPorpegateBatch(batchInputs, batchRealValues);
        }
    }
}
