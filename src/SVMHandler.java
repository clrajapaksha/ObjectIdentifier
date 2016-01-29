import java.io.IOException;

import org.opencv.core.Mat;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

public class SVMHandler {
	
	public static final int NumClass = 3;
	public static final int NumKeyPoints = 20;
	public static final int NumImgPerClass = 10;
	public static final int NumTestImgPerClass = 13;
	public static final int DescriptorSize = 32;
	
	private static svm mySVM; 
	private static svm_parameter param; 
	
	public static void init(){
		mySVM = new svm();
		param = new svm_parameter();
		
		// SVM parameters
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
	}
	public static void train(double[][] features, double[] labels){
		//training data
		svm_problem training_Data = new svm_problem();
		training_Data.l = NumClass*NumImgPerClass; // # of training examples
		training_Data.y = labels;
		//double[][] data = {{0,0}, {1,1}, {1, -5}, {-2, 1}, {-1, -8}, {5,3}};
		svm_node[][] nodes = new svm_node[labels.length][NumKeyPoints*DescriptorSize];

		// data matrix converts to libsvm format
		for(int i=0; i<labels.length; i++){
			for(int j=0; j<NumKeyPoints*DescriptorSize; j++){
				nodes[i][j] = new svm_node();
				nodes[i][j].index = j+1;
				nodes[i][j].value = features[i][j];
			}
		}

		training_Data.x = nodes;
		svm_model model = mySVM.svm_train(training_Data, param);
		try {
			mySVM.svm_save_model("model_data", model);
			System.out.println("model saved.");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("model was not saved.");
		}
	}
	
	public static double[] test(double[] data){
		double[] response = new double [NumClass+1];
		svm_model model = null;
		try {
			model = mySVM.svm_load_model("model_data");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("model was not loaded.");
		}
		
		int size = NumKeyPoints*DescriptorSize;
		svm_node[] input = new svm_node[size];
		for(int i=0; i<size; i++){
			input[i] = new svm_node();
			input[i].index = i+1;
			input[i].value = data[i];
			
		}
		double[] probabilites = new double[NumClass];
		response[0] = mySVM.svm_predict_probability(model, input, probabilites);
		for (int i=1; i<=NumClass; i++){
			response[i] = probabilites[i-1];
		}
		return response;
	}

}
