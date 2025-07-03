cwlVersion: v1.2
class: CommandLineTool
label: extract_delay_calibrator_coordinates
doc: |
    Extracts the direction of the calibrator used during delay calibration
    from the H5parm with delay calibration solutions.

baseCommand:
    - get_delay_dir.py

stdout: get_delay_dir.txt
stderr: get_delay_dir_err.txt

inputs:
    - id: delay_solutions
      label: delay_calibrator_solutions
      type: File?
      inputBinding:
        position: 0
        prefix: --h5parm
      doc: |
        Delay calibrator solutions in H5parm format.
    - id: solset
      label: solset
      type: string
      inputBinding:
        position: 1
        prefix: --solset
      doc: |
        SolutionSet to get the direction from.
      default: 'sol000'
    - id: direction_name
      label: direction_name
      type: string
      inputBinding:
        position: 0
        prefix: --direction
      doc: |
        Name of the direction to extract coordinates for.
      default: 'Dir00'

outputs:
    - id: delay_cal_dir
      label: delay_calibrator_direction
      type: File
      doc: |
        CSV table with coordinates of the delay calibrator.
      outputBinding:
        glob: delay_dir.csv

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl
