package Layers;

import Matrix.Matrix;

public class InputLayer implements Layer {
    private Matrix nodes;
    private int layerSize;

    public InputLayer(int layerSize) {
        this.layerSize = layerSize;
    }

    public Matrix feedForward(Matrix inputs) {
        this.nodes = inputs;
        return inputs;
    }


    @Override
    public void derivativeLayer(Matrix nextLayerError) {

    }

    @Override
    public Matrix getNodes() {
        return this.nodes;
    }

    @Override
    public int getLayerSize() {
        return this.layerSize;
    }

    @Override
    public void updateParameters(double alpha, int batchSize) {

    }

    @Override
    public void initGradients() {

    }
}
