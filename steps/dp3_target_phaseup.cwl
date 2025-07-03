class: CommandLineTool
cwlVersion: v1.2
id: dp3_target_phaseup
label: DP3 Target Phaseup 
doc: This tool applies the delay solutions to the target source and phase up to the various
  target directions. It appends commands to a parset created in the previous step.


baseCommand: DP3

inputs:
    - id: parset
      type: File 
      doc: Input parset file.
      default: dp3_explode.parset 
      inputBinding:
        position: 0
    - id: msin
      type: Directory
      doc: Input measurement set.
      inputBinding:
        position: 1
        prefix: msin=
        separate: false
        shellQuote: false
    - id: delay_solset
      type: File?
      doc: Input delay solution set.
      inputBinding:
        position: 2
        prefix: applycal.parmdb=
        separate: false
        shellQuote: false
    - id: max_dp3_threads
      type: int?
      default: 8
      doc: Maximum number of threads to use for DP3.
      inputBinding:
        position: 3
        prefix: numthreads=
        separate: false
        shellQuote: false

outputs:
    - id: msout
      type: Directory[]
      outputBinding:
        glob: "*.mstargetphase"
      doc: Output measurement set which has been phaseshifted, 
        averaged and had solutions applied.
    - id: logfile
      type: File
      outputBinding:
        glob: dp3_target_phaseup.log
      doc: DP3 processing log file.
    - id: errorfile
      type: File
      outputBinding:
        glob: dp3_target_phaseup_err.log
      doc: DP3 processing error log file.

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl
  - class: ResourceRequirement
    coresMin: $(inputs.max_dp3_threads)

stdout: dp3_target_phaseup.log
stderr: dp3_target_phaseup_err.log
