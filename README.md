# image tokens in words

```
./install-req.sh
./imgtokens.py "https://www.mcgawgraphics.com/cdn/shop/products/O360PF_1024x1024.jpg?v=1662584163"
```

Or you can see more than one word per token, and change the output size like:

```
./imgtokens.py "https://i0.wp.com/champagnecoloredglasses.com/wp-content/uploads/2017/06/IMGP3124.jpg?resize=2000%2C1335&ssl=1" --num-words=2 --size=1500
```
