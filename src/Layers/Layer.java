package Layers;

import Matrix.Matrix;

public interface Layer {
    public Matrix feedForward(Matrix inputs);
    public void derivativeLayer(Matrix nextLayerError);
    public Matrix getNodes();
    public int getLayerSize();
    public void updateParameters(double alpha, int batchSize);
    public void initGradients();
}
