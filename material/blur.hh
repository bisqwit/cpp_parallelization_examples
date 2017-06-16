#include <cmath>

/* blur(): Really fast O(n) gaussian blur algorithm (gaussBlur_4)
 * By Ivan Kuckir with ideas from Wojciech Jarosz
 * Adapted from http://blog.ivank.net/fastest-gaussian-blur.html
 *
 * input:  The two-dimensional array of input signal. Must contain w*h elements.
 * output: Where the two-dimensional array of blurred signal will be written
 * temp:   Another array, for temporary use. Same size as input and output.
 * w:      Width of array
 * h:      Height of array.
 * sigma:  Blurring kernel size. Must be smaller than w and h.
 * n_boxes: Controls the blurring quality. 1 = box filter. 3 = pretty good filter.
 *          Higher number = diminishingly better results, but linearly slower.
 * elem_t: Type of elements. Should be integer type.
 */
template<unsigned n_boxes, typename elem_t>
void blur(const elem_t* input, elem_t* output, elem_t* temp,
          unsigned w,unsigned h,float sigma)
{
    auto wIdeal = std::sqrt((12*sigma*sigma/n_boxes)+1);  // Ideal averaging filter width
    unsigned wl = wIdeal; if(wl%2==0) --wl;
    unsigned wu = wl+2;
    auto mIdeal = (12*sigma*sigma - n_boxes*wl*wl - 4*n_boxes*wl - 3*n_boxes)/(-4.*wl - 4);
    unsigned m = std::round(mIdeal);
    const elem_t* data = input;
    for(unsigned n=0; n<n_boxes; ++n)
    {
        unsigned r = ((n<m ? wl : wu) - 1)/2; // IDK should this be float?
        // boxBlur_4:
        float iarr = 1.f / (r+r+1);
        // boxBlurH_4 (blur horizontally for each row):
        const elem_t* scl = data; elem_t* tcl = temp;
        for(unsigned i=0; i<h; ++i)
        {
            auto ti = i*w, li = ti, ri = ti+r;
            auto fv = scl[ti], lv = scl[ti+w-1]; int val = 0;
            #pragma omp simd reduction(+:val)
            for(unsigned j=0; j<r; j++) val += scl[ti+j];
            val += (r+1)*fv;
            for(unsigned j=0  ; j<=r ; j++) { val += scl[ri++] - fv       ;   tcl[ti++] = std::round(val*iarr); }
            for(unsigned j=r+1; j<w-r; j++) { val += scl[ri++] - scl[li++];   tcl[ti++] = std::round(val*iarr); }
            for(unsigned j=w-r; j<w  ; j++) { val += lv        - scl[li++];   tcl[ti++] = std::round(val*iarr); }
        }
        // boxBlurT_4 (blur vertically for each column)
        scl = temp; tcl = output;
        for(unsigned i=0; i<w; ++i)
        {
            auto ti = i, li = ti, ri = ti+r*w;
            auto fv = scl[ti], lv = scl[ti+w*(h-1)]; int val = 0;
            #pragma omp simd reduction(+:val)
            for(unsigned j=0; j<r;  ++j) val += scl[ti + j*w];
            val += (r+1)*fv;
            for(unsigned j=0; j<=r; ++j)    { val += scl[ri] - fv     ;  tcl[ti] = std::round(val*iarr);  ri+=w; ti+=w; }
            for(unsigned j=r+1; j<h-r; ++j) { val += scl[ri] - scl[li];  tcl[ti] = std::round(val*iarr);  li+=w; ri+=w; ti+=w; }
            for(unsigned j=h-r; j<h; ++j)   { val += lv      - scl[li];  tcl[ti] = std::round(val*iarr);  li+=w; ti+=w; }
        }
        data = output;
    }
}
