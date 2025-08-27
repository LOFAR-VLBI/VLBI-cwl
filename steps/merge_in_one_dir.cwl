cwlVersion: v1.2
class: CommandLineTool
id: merge_all_in_one
doc: Merge two h5parms into one h5parm with one direction.

baseCommand: h5_merger

inputs:
  - id: first_h5
    type: File?
    doc: First h5parm
  - id: second_h5
    type: File?
    doc: Second h5parm

outputs:
    - id: merged_h5
      type: File
      doc: Merged h5parm with 1 direction
      outputBinding:
        glob: "*.onedir.h5"
    - id: logfile
      type: File[]
      doc: Log files corresponding to this step
      outputBinding:
        glob: multidir_h5_all_in_one*.log

arguments:
  - --h5_tables=$([inputs.first_h5.path, inputs.second_h5.path].join(' '))
  - --h5_out=$( inputs.second_h5.basename + '.onedir.h5' )
  - --merge_all_in_one

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $( inputs.first_h5 )
        writable: false
      - entry: $( inputs.second_h5 )
        writable: false

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: multidir_h5_all_in_one.log
stderr: multidir_h5_all_in_one_err.log
