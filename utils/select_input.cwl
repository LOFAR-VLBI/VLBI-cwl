cwlVersion: v1.2
class: ExpressionTool
doc: |
  Selects the data from the second input source if it is not null.
  Selects the data from the first input source otherwise.

inputs:
  input1: Any[]?
  input2: Any[]?

outputs:
  output: Any[]

expression: |
  ${
    return {
             "output" :
             inputs.input2 != null ? inputs.input2 : inputs.input1
           };
  }

requirements:
  InlineJavascriptRequirement: {}
