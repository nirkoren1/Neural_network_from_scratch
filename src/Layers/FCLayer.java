package Layers;

import Functions.Function;
import Matrix.Matrix;

import java.io.Serializable;

public class FCLayer implements Layer, Serializable {
    private Matrix weights;
    private Matrix gradientW;
    private Matrix gradientB;
    private Matrix biases;
    private Matrix preActFunc;
    private Matrix postActFunc;
    private Matrix inputs;
    private Function activationFunc;
    private Layer prevLayer;
    private int layerSize;

    public FCLayer(int previousLayerSize, int LayerSize, Function activationFunc, Layer prevLayer) {
        this.preActFunc = new Matrix(LayerSize, 1);
        this.postActFunc = new Matrix(LayerSize, 1);
        this.inputs = new Matrix(LayerSize, 1);
        this.activationFunc = activationFunc;

        Double scale = activationFunc.gain() / Math.sqrt(previousLayerSize);
        this.weights = Matrix.multiply(new Matrix(LayerSize, previousLayerSize), scale);

//        this.weights = Matrix.multiply(new Matrix(LayerSize, previousLayerSize), 0.1);
        this.gradientW = new Matrix(LayerSize, previousLayerSize);
        this.gradientB = new Matrix(LayerSize, 1);
        this.initGradients();
        this.biases = Matrix.multiply(new Matrix(LayerSize, 1), 0);
        this.prevLayer = prevLayer;
        this.layerSize = LayerSize;
    }

    public int getLayerSize() {
        return layerSize;
    }

    public Matrix getPreActFunc() {
        return this.preActFunc;
    }

    public Matrix getPostActFunc() {
        return postActFunc;
    }

    public Matrix getWeights() {
        return weights;
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
        this.inputs = inputs.copy();
        Matrix out = Matrix.dot(this.weights, inputs);
        out = Matrix.add(out, biases);
        this.preActFunc = out.copy();
        out = this.activationFunc.applyOn(out);
        this.postActFunc = out.copy();
        return out;
    }

    public void derivativeLayer(Matrix nextLayerError) {
        Matrix primeZ = this.preActFunc.copy();
        // this is σ'(z_j)
        Matrix sigmaTagZVec = this.activationFunc.derivativeApplyOn(primeZ);
        // σ'(z_j) * ∂C/∂a_j (next layer)
        Matrix sigmaTimesNext;
        if (sigmaTagZVec.getColumns() > 1)
            sigmaTimesNext = Matrix.dot(sigmaTagZVec, nextLayerError);
        else
            sigmaTimesNext = Matrix.multiplyElementWise(sigmaTagZVec, nextLayerError);
        // a_k * σ'(z_j) * ∂C/∂a_j (next layer)
        Matrix inputsSigmaNext = Matrix.dot(this.inputs, Matrix.transpose(sigmaTimesNext));
        // w_jk -> w_kj
        inputsSigmaNext = Matrix.transpose(inputsSigmaNext);
        // adding the current sample gradient to the batch gradient
        this.gradientW = Matrix.add(this.gradientW, inputsSigmaNext);
        this.gradientB = Matrix.add(this.gradientB, sigmaTimesNext);
        // passing the error of this layer backwards: w_jk * σ'(z_j) * ∂C/∂a_j (next layer)
        this.prevLayer.derivativeLayer(Matrix.dot(Matrix.transpose(this.weights), sigmaTimesNext));
    }

    public void updateParameters(double alpha, int batchSize) {
        this.gradientW = Matrix.multiply(this.gradientW, -1 * alpha * (1 / (double) batchSize));
        this.gradientB = Matrix.multiply(this.gradientB, -1 * alpha * (1 / (double) batchSize));
        this.weights = Matrix.add(this.weights, this.gradientW);
        this.biases = Matrix.add(this.biases, this.gradientB);
    }
}
