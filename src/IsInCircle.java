import Functions.MSE;
import Functions.Relu;
import Functions.Tanh;
import Matrix.Matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class IsInCircle {
    private static double radius = 11.0;
    private static double circleX = 0;
    private static double circleY = 0;
    private static Random random = new Random(1);
    private static List<Matrix> XTestE = new ArrayList<>();
    private static List<Matrix> YTestE = new ArrayList<>();

    private static void initTestData() {
        int testSize = 1000;
        for (int i = 0; i < testSize; i++) {
            Matrix x = new Matrix(2, 1);
            for (int j = 0; j < 2; j++) {
                x.getValues()[j][0] = -16 + 32 * random.nextDouble();
            }
            XTestE.add(x);
            Matrix y = new Matrix(1, 1);
            if (Math.pow(Math.pow(x.getValues()[0][0] - circleX, 2) + Math.pow(x.getValues()[1][0] - circleY, 2), 0.5) > radius) {
                y.getValues()[0][0] = -1.0;
            } else {
                y.getValues()[0][0] = 1.0;
            }
            YTestE.add(y);
        }
    }

    public static void writePointFile(Model model, int index) {
        double accuracy = 0;
        int testSize = 1000;
        for (int i = 0; i < testSize; i++) {
            double xPredict = model.feedForward(XTestE.get(i)).getValues()[0][0];
            double yTrue = YTestE.get(i).getValues()[0][0];
            if (xPredict * yTrue > 0) {
                accuracy += 1;
            }
        }
        accuracy /= testSize;
        System.out.println("epoch accuracy = " + accuracy);
        // write the points to a file
        Double[] xPredictsTrue = new Double[testSize];
        Double[] yPredictsTrue = new Double[testSize];
        Double[] xPredictsFalse = new Double[testSize];
        Double[] yPredictsFalse = new Double[testSize];
        for (int i = 0; i < testSize; i++) {
            if (model.feedForward(XTestE.get(i)).getValues()[0][0] > 0) {
                xPredictsTrue[i] = XTestE.get(i).getValues()[0][0];
                yPredictsTrue[i] = XTestE.get(i).getValues()[1][0];
            } else {
                xPredictsFalse[i] = XTestE.get(i).getValues()[0][0];
                yPredictsFalse[i] = XTestE.get(i).getValues()[1][0];
            }
        }
        PointsWriter pointsWriter = new PointsWriter(xPredictsTrue, yPredictsTrue, testSize, "predicted-true-" + index);
        pointsWriter.writePoints();
        pointsWriter = new PointsWriter(xPredictsFalse, yPredictsFalse, testSize, "predicted-false-" + index);
        pointsWriter.writePoints();
    }

    public static void main(String[] args) {
        initTestData();
        List<Matrix> X = new ArrayList<>();
        List<Matrix> Y = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            Matrix x = new Matrix(2, 1);
            for (int j = 0; j < 2; j++) {
                x.getValues()[j][0] = -16 + 32 * random.nextDouble();
            }
            X.add(x);
            Matrix y = new Matrix(1, 1);
            if (Math.pow(Math.pow(x.getValues()[0][0] - circleX, 2) + Math.pow(x.getValues()[1][0] - circleY, 2), 0.5) > radius) {
                y.getValues()[0][0] = -1.0;
            } else {
                y.getValues()[0][0] = 1.0;
            }
            Y.add(y);
        }

        List<Matrix> XTest = new ArrayList<>();
        List<Matrix> YTest = new ArrayList<>();
        double accuracy = 0;
        int testSize = 1000;
        for (int i = 0; i < testSize; i++) {
            Matrix x = new Matrix(2, 1);
            for (int j = 0; j < 2; j++) {
                x.getValues()[j][0] = -16 + 32 * random.nextDouble();
            }
            XTest.add(x);
            Matrix y = new Matrix(1, 1);
            if (Math.pow(Math.pow(x.getValues()[0][0] - circleX, 2) + Math.pow(x.getValues()[1][0] - circleY, 2), 0.5) > radius) {
                y.getValues()[0][0] = -1.0;
            } else {
                y.getValues()[0][0] = 1.0;
            }
            YTest.add(y);
        }

        Model model = new Model(64, new MSE(), 0.01);
        model.add(2, new Relu());
        model.add(16, new Relu());
        model.add(32, new Tanh());
//        model.add(64, new Sigmoid());
//        model.add(32, new Sigmoid());
        model.add(16, new Tanh());
        model.add(1, new Tanh());

        model.trainModel(X, Y, 30, XTest, YTest, 1);
        System.out.println("----------TEST-----------");


        double avgErrorRadius = 0;
        int numOfError = 0;

        for (int i = 0; i < testSize; i++) {
            double xPredict = model.feedForward(XTest.get(i)).getValues()[0][0];

            double yTrue = YTest.get(i).getValues()[0][0];
            if (xPredict * yTrue > 0) {
                accuracy += 1;
            }
            else {
                avgErrorRadius += Math.pow(Math.pow(XTest.get(i).getValues()[0][0] - circleX, 2) + Math.pow(XTest.get(i).getValues()[1][0] - circleY, 2), 0.5);
                numOfError += 1;
            }
        }
        accuracy /= testSize;
        avgErrorRadius /= numOfError;
        System.out.println("Total accuracy = " + accuracy + "  avg radius of error = " + avgErrorRadius);
        // write the points to a file
        Double[] xPredictsTrue = new Double[testSize];
        Double[] yPredictsTrue = new Double[testSize];
        Double[] xPredictsFalse = new Double[testSize];
        Double[] yPredictsFalse = new Double[testSize];
        for (int i = 0; i < testSize; i++) {
            if (model.feedForward(XTest.get(i)).getValues()[0][0] > 0) {
                xPredictsTrue[i] = XTest.get(i).getValues()[0][0];
                yPredictsTrue[i] = XTest.get(i).getValues()[1][0];
            } else {
                xPredictsFalse[i] = XTest.get(i).getValues()[0][0];
                yPredictsFalse[i] = XTest.get(i).getValues()[1][0];
            }
        }
        PointsWriter pointsWriter = new PointsWriter(xPredictsTrue, yPredictsTrue, testSize, "predicted-true.txt");
        pointsWriter.writePoints();
        pointsWriter = new PointsWriter(xPredictsFalse, yPredictsFalse, testSize, "predicted-false.txt");
        pointsWriter.writePoints();
    }
}
