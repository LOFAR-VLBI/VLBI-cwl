class: CommandLineTool
cwlVersion: v1.2
id: concatfiles
label: Concatenate files
doc: |
    Takes an array of text files and concatenates them.
    The output file can be given a file name and, optionally,
    a suffix.

baseCommand: cat

stdout: $(inputs.file_prefix).$(inputs.file_suffix)

inputs:
    - id: file_list
      type: File[]
      inputBinding:
        position: 0
      doc: The list of files to be concatenated.

    - id: file_prefix
      type: string
      doc: The output file name.

    - id: file_suffix
      type: string?
      default: log
      doc: The output file extension.

outputs:
    - id: output
      type: stdout
      doc: The concatenated file.
