class: CommandLineTool
cwlVersion: v1.2
id: subtract_with_wsclean
label: Subtract with WSClean
doc: This step uses WSClean to subtract visibilities corresponding to model images.

baseCommand: subtract_with_wsclean

inputs:
  - id: msin
    type: Directory
    doc: MeasurementSet for source subtraction.
    inputBinding:
      prefix: "--mslist"
      position: 1

  - id: model_image_folder
    type: Directory
    doc: Directory containing 1.2" (or optionally other resolution) model images.
    inputBinding:
      prefix: "--model_image_folder"
      position: 2

  - id: facet_regions
    type: File
    doc: The DS9 region file that defines the facets for prediction.
    inputBinding:
      prefix: "--facets_predict"
      position: 3

  - id: h5parm
    type: File
    doc: The HDF5 solution file containing the solutions for prediction.
    inputBinding:
      prefix: "--h5parm_predict"
      position: 4

  - id: copy_to_local_scratch
    type: boolean?
    default: false
    inputBinding:
      prefix: "--copy_to_local_scratch"
      position: 5
      separate: false
    doc: Specific option for using --bypass-file-store on the Spider cluster to run predict and subtract on local scratch.

  - id: ncpu
    type: int?
    doc: Number of cores to use during the subtract.
    default: 16

outputs:
  - id: logfile
    type: File[]
    doc: Log files from current step.
    outputBinding:
      glob: subtract_fov*.log
  - id: subtracted_ms
    type: Directory
    doc: MS subtracted data
    outputBinding:
      glob: subfov_*.ms

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing: >
      ${
        // Set 'writable' on the "msin" entry only if copy_to_local_scratch is true.
        let stagedListing = [
          { entry: inputs.msin },
          { entry: inputs.model_image_folder },
          { entry: inputs.facet_regions },
          { entry: inputs.h5parm }
        ];
        if (!inputs.copy_to_local_scratch) {
          stagedListing[0].writable = true;
        }
        return stagedListing;
      }

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl
  - class: ResourceRequirement
    coresMin: $(inputs.ncpu)

stdout: subtract_fov.log
stderr: subtract_fov_err.log
