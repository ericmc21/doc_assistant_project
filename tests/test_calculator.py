import pytest
from unittest.mock import MagicMock

from tools import ToolLogger, create_calculator_tool


@pytest.fixture
def mock_logger():
    return MagicMock(spec=ToolLogger)


@pytest.fixture
def calculator(mock_logger):
    return create_calculator_tool(mock_logger)


class TestCalculatorSafety:
    def test_addition(self, calculator):
        result = calculator.invoke({"expression": "1 + 2"})
        assert "3" in result

    def test_subtraction(self, calculator):
        result = calculator.invoke({"expression": "10 - 4"})
        assert "6" in result

    def test_multiplication(self, calculator):
        result = calculator.invoke({"expression": "5 * 6"})
        assert "30" in result

    def test_division(self, calculator):
        result = calculator.invoke({"expression": "20 / 4"})
        assert "5" in result

    def test_parentheses(self, calculator):
        result = calculator.invoke({"expression": "(2 + 3) * 4"})
        assert "20" in result

    def test_decimal(self, calculator):
        result = calculator.invoke({"expression": "1.5 + 2.5"})
        assert "4" in result

    def test_rejects_alpha_characters(self, calculator):
        result = calculator.invoke({"expression": "2 + a"})
        assert "Error" in result

    def test_rejects_dunder_import(self, calculator):
        result = calculator.invoke({"expression": "__import__('os').system('ls')"})
        assert "Error" in result

    def test_rejects_exec(self, calculator):
        result = calculator.invoke({"expression": "exec('import os')"})
        assert "Error" in result

    def test_rejects_semicolon(self, calculator):
        result = calculator.invoke({"expression": "1+1; import os"})
        assert "Error" in result

    def test_rejects_square_brackets(self, calculator):
        result = calculator.invoke({"expression": "[1,2,3]"})
        assert "Error" in result

    def test_empty_expression_rejected(self, calculator):
        result = calculator.invoke({"expression": ""})
        assert "Error" in result

    def test_division_by_zero_returns_error(self, calculator):
        result = calculator.invoke({"expression": "1 / 0"})
        assert "Error" in result


class TestCalculatorLogging:
    def test_logs_successful_calculation(self, calculator, mock_logger):
        calculator.invoke({"expression": "2 + 2"})
        mock_logger.log_tool_use.assert_called_once()
        tool_name, inputs, output = mock_logger.log_tool_use.call_args[0]
        assert tool_name == "calculator"
        assert inputs["expression"] == "2 + 2"
        assert "result" in output

    def test_logs_validation_failure(self, calculator, mock_logger):
        """Rejected expressions must be logged, not silently discarded."""
        calculator.invoke({"expression": "import os"})
        mock_logger.log_tool_use.assert_called_once()
        tool_name, inputs, output = mock_logger.log_tool_use.call_args[0]
        assert tool_name == "calculator"
        assert "error" in output

    def test_logs_runtime_error(self, calculator, mock_logger):
        calculator.invoke({"expression": "1 / 0"})
        mock_logger.log_tool_use.assert_called_once()
        _, _, output = mock_logger.log_tool_use.call_args[0]
        assert "error" in output

    def test_result_format(self, calculator, mock_logger):
        result = calculator.invoke({"expression": "10 + 5"})
        assert result.startswith("Result:")
        assert "15" in result
