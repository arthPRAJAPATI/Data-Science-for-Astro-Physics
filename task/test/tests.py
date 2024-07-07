from hstest import *
from math import isclose


def is_float(num: str):
    try:
        float(num)
        return True
    except ValueError:
        return False


class Stage3Test(PlottingTest):
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
        self.check_outputs_number(3, user_output)

        # check fraction of galaxies with n > 2 for CGs galaxies
        cgs_fraction = [0.6394557823129252]
        user_values = [float(value) for value in user_output.split()][:1]
        self.check_num_values(cgs_fraction, user_values,
                              "The fraction of galaxies with Sersic index n > 2 for groups galaxies is wrong.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=1.0e-2)

        # check fraction of galaxies with n > 2 for isolated galaxies
        p_values = [0.29245283018867924]
        user_values = [float(value) for value in user_output.split()][1:2]
        self.check_num_values(p_values, user_values,
                              "The fraction of galaxies with Sersic index n > 2 for isolated galaxies is wrong.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=1.0e-2)

        # check Kolmogorov-Smirnov test p-value
        p_values = [4.163336342344337e-14]
        user_values = [float(value) for value in user_output.split()][2:3]
        self.check_num_values(p_values, user_values, "The p-value of Kolmogorov-Smirnov test is wrong.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=5.0e-2)

        fanswers = open("answers.txt", "w+")
        fanswers.write("""1. The concentration of early-type galaxies is greater in groups of galaxies.

2. No. The resulting p-value indicates that we reject the null hypothesis\nthat the two given samples are drawn from the same probability distribution.""")
        fanswers.close()

        return CheckResult.correct()


if __name__ == '__main__':
    Stage3Test().run_tests()
