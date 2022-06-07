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

        Model model = new Model(16, new MSE(), 0.05);
        model.add(2, new Relu());
        model.add(16, new Relu());
        model.add(32, new Sigmoid());
        model.add(32, new Sigmoid());
        model.add(16, new Sigmoid());
        model.add(1, new Sigmoid());

        model.trainModel(X, Y, 500);
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
                System.out.println(YTest.get(i));
                System.out.println(xPredict);
                System.out.println("---------------------");
            }
        }
        accuracy /= testSize;
        System.out.println("Total accuracy = " + accuracy);
    }
}
