from hstest import *
from math import isclose


def is_float(num: str):
    try:
        float(num)
        return True
    except ValueError:
        return False


class Stage4Test(PlottingTest):
    def check_outputs_number(self, values_number: int, user_output: str):
        outputs = user_output.split()

        if not all(is_float(output) for output in outputs):
            raise WrongAnswer(f"Answer '{user_output}' contains non-numeric values.")

        if len(outputs) != values_number:
            raise WrongAnswer(f"Answer contains {len(outputs)} values, but {values_number} values are expected.")

    def check_num_values(self, values: list, user_values: list, message: str, rel_tol=1.0e-3):
        if not all(isclose(value, user_value, rel_tol=rel_tol) for value, user_value in zip(values, user_values)):
            raise WrongAnswer(message)

    @dynamic_test
    def test(self):
        pr = TestedProgram()
        user_output = pr.start().strip()

        if len(user_output.strip()) == 0:
            raise WrongAnswer("Seems like your program does not show any output.")

        # check output format
        self.check_outputs_number(5, user_output)

        # check Shapiro tests p-values
        p_value = [0.8418697118759155]
        user_values = [float(value) for value in user_output.split()][:1]
        self.check_num_values(p_value, user_values,
                              "The p-value of Shapiro-Wilk test for the mean surface brightness of the IGL is wrong.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=1.0e-2)

        p_value = [0.07041092216968536]
        user_values = [float(value) for value in user_output.split()][1:2]
        self.check_num_values(p_value, user_values,
                              "The p-value of Shapiro-Wilk test for mean Sersic index is wrong.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=1.0e-2)

        p_value = [0.5136213898658752]
        user_values = [float(value) for value in user_output.split()][2:3]
        self.check_num_values(p_value, user_values,
                              "The p-value of Shapiro-Wilk test for mean numerical galaxy type is wrong.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=1.0e-2)

        # check non-correlation tests p-values
        p_value = [0.0003025740445687669]
        user_values = [float(value) for value in user_output.split()][3:4]
        self.check_num_values(p_value, user_values,
                              "The p-value of non-correlation test between the mean surface brightness of the IGL and mean Sersic indexes is wrong.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=1.0e-2)

        p_value = [5.310877708860126e-05]
        user_values = [float(value) for value in user_output.split()][4:5]
        self.check_num_values(p_value, user_values,
                              "The p-value of non-correlation tests between the mean surface brightness of the IGL and mean numerical galaxy type is wrong.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=1.0e-2)

        fanswers = open("answers.txt", "w+")
        fanswers.write("""We can see correlation:
1. Lower value of mean Sersic index corresponds to higher IGL mean surface\nbrightness. 
2. Higher value of mean numerical Hubble stage corresponds to higher IGL mean\nsurface brightness. 

However, it is important to note that the relationship between IGL and mean\nSersic index or Hubble stages can be complex and may vary depending on the\nspecific group of galaxies being studied. """)
        fanswers.close()

        return CheckResult.correct()


if __name__ == '__main__':
    Stage4Test().run_tests()
