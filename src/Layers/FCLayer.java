package Layers;

import Functions.Function;
import Matrix.Matrix;

public class FCLayer implements Layer {
    private Matrix weights;
    private Matrix gradientW;
    private Matrix gradientB;
    private Matrix biases;
    private Matrix nodes;
    private Function activationFunc;
    private Layer prevLayer;
    private int layerSize;

    public FCLayer(int previousLayerSize, int LayerSize, Function activationFunc, Layer prevLayer) {
        this.nodes = new Matrix(LayerSize, 1);
        this.activationFunc = activationFunc;
        this.weights = new Matrix(LayerSize, previousLayerSize);
        this.gradientW = new Matrix(LayerSize, previousLayerSize);
        this.gradientB = new Matrix(LayerSize, 1);
        this.initGradients();
        this.biases = new Matrix(LayerSize, 1);
        this.prevLayer = prevLayer;
        this.layerSize = LayerSize;
    }

    public int getLayerSize() {
        return layerSize;
    }

    public Matrix getNodes() {
        return this.nodes;
    }

    public void initGradients() {
        for (int i = 0; i < this.gradientW.getRows(); i++) {
            for (int j = 0; j < this.gradientW.getColumns(); j++) {
                this.gradientW.getValues()[i][j] = 0.0;
            }
        }
        for (int i = 0; i < this.gradientB.getRows(); i++) {
            for (int j = 0; j < this.gradientB.getColumns(); j++) {
                this.gradientB.getValues()[i][j] = 0.0;
            }
        }
    }

    public Matrix feedForward(Matrix inputs) {
        Matrix out = Matrix.dot(this.weights, inputs);
        out = Matrix.add(out, biases);
        this.nodes = Matrix.add(out, biases);
        for (int i = 0; i < out.getRows(); i++) {
            for (int j = 0; j < out.getColumns(); j++) {
                out.getValues()[i][j] = this.activationFunc.applyOn(out.getValues()[i][j]);
            }
        }
        return out;
    }

    public void derivativeLayer(Matrix nextLayerError) {
        Double[] primeZ = new Double[this.nodes.getRows()];
        for (int i = 0; i < this.nodes.getRows(); i++) {
            primeZ[i] = this.activationFunc.derivativeApplyOn(this.nodes.getValues()[i][0]);
        }
        Matrix primeZMatrix = Matrix.diagonalizeVector(primeZ);
        for (int i = 0; i < this.weights.getRows(); i++) {
            for (int j = 0; j < this.weights.getColumns(); j++) {
                Matrix E = Matrix.getEMatrix(this.weights.getRows(), this.weights.getColumns(), i, j);
                this.gradientW.getValues()[i][j] += Matrix.dot(nextLayerError, Matrix.dot(primeZMatrix, Matrix.dot(E,
                                                                this.prevLayer.getNodes()))).getValues()[0][0];
            }
        }
        for (int i = 0; i < this.biases.getRows(); i++) {
            Matrix E = Matrix.getEMatrix(this.nodes.getRows(), this.nodes.getColumns(), i, 0);
            this.gradientB.getValues()[i][0] += Matrix.dot(nextLayerError, Matrix.dot(primeZMatrix, E)).getValues()[0][0];
        }
        this.prevLayer.derivativeLayer(Matrix.dot(nextLayerError, Matrix.dot(primeZMatrix, this.weights)));
    }

    public void updateParameters(double alpha, int batchSize) {
        this.gradientW = Matrix.multiply(this.gradientW, -1 * alpha * (1 / (double) batchSize));
        this.gradientB = Matrix.multiply(this.gradientB, -1 * alpha * (1 / (double) batchSize));
        this.weights = Matrix.add(this.weights, this.gradientW);
        this.biases = Matrix.add(this.biases, this.gradientB);
    }
}
