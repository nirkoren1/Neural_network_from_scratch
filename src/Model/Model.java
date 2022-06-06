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
    private double discount;
    private Random rand = new Random();

    public Model(int batchSize, CostFunction costFunction, double discount) {
        this.layers = new ArrayList<>();
        this.batchSize = batchSize;
        this.costFunction = costFunction;
        this.discount = discount;
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
        Matrix result = inputs;
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
            layer.updateParameters(this.discount, this.batchSize);
        }
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
            this.backPorpegateBatch(batchInputs, batchRealValues);
            System.out.println("epoch: " + j);
        }
    }
}
