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
    public Matrix getPreActFunc() {
        return this.nodes;
    }

    @Override
    public Matrix getPostActFunc() {return this.nodes;}

    @Override
    public Matrix getWeights() {return new Matrix(0, 0);}

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
