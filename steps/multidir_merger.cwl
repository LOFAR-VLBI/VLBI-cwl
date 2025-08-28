cwlVersion: v1.2
class: CommandLineTool
id: multidir_h5_merger
doc: Merges multiple h5parms from containing directions into a single multi-directional h5parm.

baseCommand: h5_merger

inputs:
  - id: h5parms
    type: File[]
    doc: Input h5parms
    inputBinding:
      prefix: "-in"
      position: 1
      itemSeparator: " "
      separate: true

outputs:
    - id: multidir_h5
      type: File
      doc: Merged multi-directional h5parm.
      outputBinding:
        glob: merged.h5
    - id: logfile
      type: File[]
      doc: Log files corresponding to this step.
      outputBinding:
        glob: multidir_h5_merger*.log

arguments:
  - --h5_out=merged.h5

requirements:
  - class: InlineJavascriptRequirement

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: multidir_h5_merger.log
stderr: multidir_h5_merger_err.log
