Group Members:

Sudireddy Raghavender Reddy (A20554654)
Chaitanya Durgesh Nynavarapu(A20561894)
Purnachandra Reddy Peddasura(A20544751)
jeswanth jayavarapu A20547505
-------------------------------------------------------------------

Instructions to run code 

go to project folder location in Comand prompt (Terminal) and type

python -m venv myenv

myenv\Scripts\activate

pip install -r requirements.txt

cd LassoHomotopy

pytest tests/

--------------------------------------------------------------------

to run example usage

cd ..

python example_usage.py

-----------------------------------------------------------------------

we have tested the model on additional test cases
test_sparsity
test_high_alpha_zeroes_out
test_comparison_with_sklearn
test_edge_case_single_feature

---------------------------------------------------------------------------------------------------------------------------
What does the model you have implemented do and when should it be used?

The model implements Lasso Regression using the Homotopy Method, which efficiently finds sparse solutions by shrinking some coefficients to zero. It is useful for feature selection and handling high-dimensional datasets such as machine learning tasks in genomics or text analysis.

How did you test your model to determine if it is working reasonably correctly?

The model was tested using pytest unit tests to check if it fits data correctly and produces predictions of the expected shape. Additionally, Jupyter Notebook visualizations were used to compare predicted vs. actual values and analyze the effect of regularization on coefficients.

What parameters have you exposed to users of your implementation in order to tune performance?

Users can adjust reg_strength to control regularization, max_steps for iteration limits, and threshold for convergence precision. These parameters allow fine-tuning of sparsity versus fit and computational efficiency.

Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

The model may struggle with highly correlated features, very large datasets, and unscaled input data. These issues could be eliminated by feature preprocessing, alternative optimization methods.
