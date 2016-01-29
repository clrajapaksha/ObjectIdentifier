import java.io.IOException;

import libsvm.*;
public class SVM_ {

	public void myMethod(){
		svm mySVM = new svm();
		// SVM parameters
		svm_parameter param = new svm_parameter();
		/*params.svm_type = svm_parameter.C_SVC;
		params.kernel_type = svm_parameter.RBF;
		params.gamma = 0.1;
		params.C = 0.001;*/
		
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = svm_parameter.POLY;
		param.degree = 3;
		param.gamma = 0.1;
		param.coef0 = 0.1;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 0.4;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 1;
		param.nr_weight = 0;
		param.weight_label = null;
		param.weight = null;
		
		//training data
		svm_problem training_Data = new svm_problem();
		training_Data.l = 6; // # of training examples
		double[] classes = {1, 1, -1, 1, -1, -1}; 
		training_Data.y = classes;
		double[][] data = {{0,0}, {1,1}, {1, -5}, {-2, 1}, {-1, -8}, {5,3}};
		svm_node[][] nodes = new svm_node[6][2];
		
		// data matrix converts to libsvm format
		for(int i=0; i<data.length; i++){
			for(int j=0; j<2; j++){
				nodes[i][j] = new svm_node();
				nodes[i][j].index = j+1;
				nodes[i][j].value = data[i][j];
			}
		}
		
		training_Data.x = nodes;
		
		// svm_problem,svm_parameter return svm_model 
		svm_model model = mySVM.svm_train(training_Data, param);
		try {
			mySVM.svm_save_model("model", model);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("model was not saved.");
		}
		svm_model lmodel = null;
		try {
			lmodel = mySVM.svm_load_model("model");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("model was not loaded.");
		}
		//model and node[] returns double
		svm_node[] input = new svm_node[2];
		input[0] = new svm_node();
		input[0].index = 1;
		input[0].value = 0;
		input[0] = new svm_node();
		input[0].index = 2;
		input[0].value = -10;
		double response = mySVM.svm_predict(lmodel, input);
		double[] prob = new double[2];
		double pResponse = mySVM.svm_predict_probability(lmodel, input, prob);
		System.out.println("Output = "+response);
		System.out.println("prob output = "+pResponse);
		System.out.println("1: "+prob[0]+"   -1: "+prob[1]);
	}
}
