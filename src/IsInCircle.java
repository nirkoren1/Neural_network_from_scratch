import Functions.MSE;
import Functions.Relu;
import Functions.Sigmoid;
import Matrix.Matrix;
import Model.Model;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class IsInCircle {
    private static Random random = new Random();
    public static void main(String[] args) {
        List<Matrix> X = new ArrayList<>();
        List<Matrix> Y = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            Matrix x = new Matrix(2, 1);
            for (int j = 0; j < 2; j++) {
                x.getValues()[j][0] = -16 + 32 * random.nextDouble();
            }
            X.add(x);
            Matrix y = new Matrix(1, 1);
            if (Math.pow(Math.pow(x.getValues()[0][0], 2) + Math.pow(x.getValues()[1][0], 2), 0.5) > 9.0) {
                y.getValues()[0][0] = -1.0;
            } else {
                y.getValues()[0][0] = 1.0;
            }
            Y.add(y);
        }

        Model model = new Model(64, new MSE(), 0.05);
        model.add(2, new Relu());
        model.add(16, new Relu());
        model.add(32, new Sigmoid());
//        model.add(64, new Sigmoid());
//        model.add(32, new Sigmoid());
        model.add(16, new Sigmoid());
        model.add(1, new Sigmoid());

        model.trainModel(X, Y, 50);
        System.out.println("----------TEST-----------");
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
            if (Math.pow(Math.pow(x.getValues()[0][0], 2) + Math.pow(x.getValues()[1][0], 2), 0.5) > 9.0) {
                y.getValues()[0][0] = -1.0;
            } else {
                y.getValues()[0][0] = 1.0;
            }
            YTest.add(y);
        }

        double avgErrorRadius = 0;
        int numOfError = 0;

        for (int i = 0; i < testSize; i++) {
//            System.out.println(XTest.get(i));
//            System.out.println(YTest.get(i));
//            System.out.println(model.feedForward(XTest.get(i)));
//            System.out.println("---------------------");
            double xPredict = model.feedForward(XTest.get(i)).getValues()[0][0];
            double yTrue = YTest.get(i).getValues()[0][0];
            if ((xPredict < 0 && yTrue < 0) || (xPredict > 0 && yTrue > 0)) {
                accuracy += 1;
            } else {
                System.out.println(XTest.get(i));
                avgErrorRadius += Math.pow(Math.pow(XTest.get(i).getValues()[0][0], 2) + Math.pow(XTest.get(i).getValues()[1][0], 2), 0.5);
                numOfError += 1;
                System.out.println(YTest.get(i));
                System.out.println(xPredict);
                System.out.println("---------------------");
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
