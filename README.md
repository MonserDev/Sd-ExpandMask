# Sd-ExpandMask
ExpandMask Extention for Stablediffusion WebUI

[img](https://i.imgur.com/mWE7wnd.png[/img])

### New Expand script

```python
def dilate_mask(mask, dilation_amt):
                        mask = np.array(mask.convert("L"))
                        x, y = np.meshgrid(np.arange(dilation_amt), np.arange(dilation_amt))
                        center = dilation_amt // 2
                        dilation_kernel = ((x - center)**2 + (y - center)**2 <= center**2).astype(np.uint8)
                        dilated_binary_img = binary_dilation(mask, dilation_kernel)
                        dilated_mask = Image.fromarray(dilated_binary_img.astype(np.uint8) * 255)
                        return dilated_mask
```
