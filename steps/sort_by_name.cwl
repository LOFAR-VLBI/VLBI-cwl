cwlVersion: v1.2
class: CommandLineTool
requirements:
  - class: InlineJavascriptRequirement

label: Sort input files by basename
doc: |
  Takes a File[] input, sorts it alphabetically by filename, and returns
  the sorted list of files.

inputs:
  input_files:
    type:
      - File[]?
  input_directories:
    type:
      - Directory[]?

outputs:
  sorted_files:
    type:
      - File[]?
    outputBinding:
      outputEval: |
        ${
          if (inputs.input_files) {
            var sorted = inputs.input_files.sort(function(a, b) {
              return a.basename.localeCompare(b.basename);
            });
            return sorted;
          } else {
            return null;
          }
        }
  sorted_directories:
    type:
      - Directory[]?
    outputBinding:
      outputEval: |
        ${
          if (inputs.input_directories) {
            var sorted = inputs.input_directories.sort(function(a, b) {
              return a.basename.localeCompare(b.basename);
            });
            return sorted;
          } else {
            return null;
          }
        }

baseCommand: echo
stdout: sort_files_by_name.txt
stderr: sort_files_by_name_err.txt

