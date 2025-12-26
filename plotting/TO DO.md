# TO DO

Just some notes on what I would like to do and how I think it might be done. In no particualr order

## Error handling and making fewer assumptions about the quality of user input

It would be good for the plotting drivers to double-check that they have been given sensible inputs (eg no mixing of prebinned and unbinned datasets for plot_histogram())

And when things do break, it would be nice to have readable error messages, instead of just whatever natural python type errors or attribute access errors or whatever

 - Check that inputs to plotting drivers are reasonable
 - Check if covariance matrices really are the expected shape
 - Check if prebinned histograms really are the expected shape
 - ...

## Labels for prebinned data

Need a scheme for intelligently setting y-axis (in plot_histogram()) and colorbar (in draw_matrix() and draw_radial_histogram()) labels for prebinned data. This should inteligently handle when there has been jacobian or per-block normalizations...

## Handling non-square covariance matrices

Need to track variances along each axis

## Matrices with interesting bin labels

Via a binning() class??

## Rotated fancy prebinned axis labels overlapping with big axis labels

It must be possible to detect this and offset the axis labels

## Better automatic sym detection in draw_matrix() 

Need to take into account what is being plotted (covariance matrices should always have sym=True, ...)

Might want to be a function of the variable key? 

## Automatic logc setting in draw_matrix()

Need to think a bit about what would be desireable here

## Show flow bins sensibly

## Prevent multipliers above the axis, which clashes with the CMS label text

This is easy I think, just need to google it when I'm back in internet :)

I've done something, not sure if it fixes. Will revisit if/when I reproduce the problem naturally.