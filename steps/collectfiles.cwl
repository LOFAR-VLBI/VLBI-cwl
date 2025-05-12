class: CommandLineTool
cwlVersion: v1.2
id: collectfiles
label: Collect files
doc: |
    This step stores a file or directory
    (or an array of files or directories)
    in separate directory.

baseCommand: echo
arguments: ["Collecting files in", $(inputs.sub_directory_name)]

inputs:
  - id: files
    type:
      - File
      - Directory
      - type: array
        items:
           - File
           - Directory
    doc: |
        The files or directories that should be placed
        in the output directory.

  - id: sub_directory_name
    type: string
    doc: |
        A string that determines
        the name of the output directory.

outputs:
  - id: dir
    type: Directory
    outputBinding:
        glob: $(runtime.outdir)
        outputEval: |
          ${
            self[0].basename = inputs.sub_directory_name;
            return self;
          }

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.files)
        writable: true  # Needed to prevent CWL from generating symlinks
