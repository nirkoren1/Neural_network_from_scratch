import Functions.MSE;
import Functions.Relu;
import Functions.Sigmoid;
import Matrix.Matrix;
import Model.Model;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

public class XorEx {
    private static Random random = new Random();
    public static void main(String[] args) {
        List<Matrix> X = new ArrayList<>();
        List<Matrix> Y = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            Matrix x = new Matrix(2, 1);
            for (int j = 0; j < 2; j++) {
                x.getValues()[j][0] = (double) random.nextInt(2);
            }
            X.add(x);
            Matrix y = new Matrix(1, 1);
            if (Math.abs(x.getValues()[0][0] - x.getValues()[1][0]) == 0.0) {
                y.getValues()[0][0] = -1.0;
            } else {
                y.getValues()[0][0] = 1.0;
            }
            Y.add(y);
        }

        Model model = new Model(16, new MSE(), 0.05);
        model.add(2, new Relu());
        model.add(8, new Relu());
        model.add(16, new Sigmoid());
        model.add(1, new Sigmoid());

        model.trainModel(X, Y, 1000);
        System.out.println("----------TEST-----------");
        List<Matrix> XTest = new ArrayList<>();
        List<Matrix> YTest = new ArrayList<>();
        double totalError = 0;
        int testSize = 20;
        for (int i = 0; i < testSize; i++) {
            int rand = random.nextInt(1000);
            XTest.add(X.get(rand));
            YTest.add(Y.get(rand));
        }
        for (int i = 0; i < testSize; i++) {
            System.out.println(XTest.get(i));
            System.out.println(YTest.get(i));
            System.out.println(model.feedForward(XTest.get(i)));
            System.out.println("---------------------");
            totalError += Math.abs(model.feedForward(XTest.get(i)).getValues()[0][0] - YTest.get(i).getValues()[0][0]);
        }
        totalError /= testSize;
        System.out.println("Total error = " + totalError);
    }
}
