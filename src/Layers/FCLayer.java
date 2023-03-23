package Layers;

import Functions.Function;
import Matrix.Matrix;

public class FCLayer implements Layer {
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
        this.weights = Matrix.multiply(new Matrix(LayerSize, previousLayerSize), 0.1);
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
        for (int i = 0; i < out.getRows(); i++) {
            for (int j = 0; j < out.getColumns(); j++) {
                out.getValues()[i][j] = this.activationFunc.applyOn(out.getValues()[i][j]);
            }
        }
        this.postActFunc = out.copy();
        return out;
    }

    public void derivativeLayer(Matrix nextLayerError) {
        Double[][] primeZ = new Double[this.preActFunc.getRows()][1];
        for (int i = 0; i < this.preActFunc.getRows(); i++) {
            primeZ[i][0] = this.activationFunc.derivativeApplyOn(this.preActFunc.getValues()[i][0]);
        }
        // this is σ'(z_j)
        Matrix sigmaTagZVec = new Matrix(this.preActFunc.getRows(), 1, primeZ);
        // σ'(z_j) * ∂C/∂a_j (next layer)
        Matrix sigmaTimesNext = Matrix.dotElementWise(sigmaTagZVec, nextLayerError);
        // a_k * σ'(z_j) * ∂C/∂a_j (next layer)
        Matrix inputsSigmaNext = Matrix.dot(this.inputs, Matrix.trancePose(sigmaTimesNext));
        // w_jk -> w_kj
        inputsSigmaNext = Matrix.trancePose(inputsSigmaNext);
        this.gradientW = Matrix.add(gradientW, inputsSigmaNext);
//        for (int i = 0; i < this.weights.getRows(); i++) {
//            for (int j = 0; j < this.weights.getColumns(); j++) {
//                Matrix E = Matrix.getEMatrix(this.weights.getRows(), this.weights.getColumns(), i, j);
//                this.gradientW.getValues()[i][j] += Matrix.dot(nextLayerError, Matrix.dot(primeZMatrix, Matrix.dot(E,
//                                                                this.prevLayer.getPreActFunc()))).getValues()[0][0];
//            }
//        }
//        for (int i = 0; i < this.biases.getRows(); i++) {
//            Matrix E = Matrix.getEMatrix(this.preActFunc.getRows(), this.preActFunc.getColumns(), i, 0);
//            this.gradientB.getValues()[i][0] += Matrix.dot(nextLayerError, Matrix.dot(primeZMatrix, E)).getValues()[0][0];
//        }
        this.gradientB = Matrix.add(this.gradientB, sigmaTimesNext);
        // passing the error of this layer backwards: w_jk *
        this.prevLayer.derivativeLayer(Matrix.dot(Matrix.trancePose(this.weights), sigmaTimesNext));
    }

    public void updateParameters(double alpha, int batchSize) {
        this.gradientW = Matrix.multiply(this.gradientW, -1 * alpha * (1 / (double) batchSize));
        this.gradientB = Matrix.multiply(this.gradientB, -1 * alpha * (1 / (double) batchSize));
        this.weights = Matrix.add(this.weights, this.gradientW);
        this.biases = Matrix.add(this.biases, this.gradientB);
    }
}
