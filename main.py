from pathlib import Path

from style_transfer_wrappers import AdaINStyleTransfer, MicroASTStyleTransfer


def main():
    adain = AdaINStyleTransfer()
    microast = MicroASTStyleTransfer()

    content_img = Path("test_adain/glu.jpeg")
    style_img_1 = Path("pytorch_AdaIN/input/style/woman_with_hat_matisse.jpg")
    style_img_2 = Path("pytorch_AdaIN/input/style/la_muse.jpg")

    for name, model in (("adain", adain), ("microast", microast)):
        # Cas 1 : 1 image de contenu + 1 image de style
        out_single = model.transfer(
            content=content_img,
            style=style_img_1,
            alpha=0.8,
            size=512,
            output_path=f"outputs/{name}_1style.png",
        )
        print(f"[{name}] 1 style -> {out_single.size} (outputs/{name}_1style.png)")

        # Cas 2 : 1 image de contenu + 2 images de style (mélange 50/50)
        out_double = model.transfer(
            content=content_img,
            style=[style_img_1, style_img_2],
            style_weight=0.5,
            alpha=0.8,
            size=512,
            output_path=f"outputs/{name}_2styles.png",
        )
        print(f"[{name}] 2 styles -> {out_double.size} (outputs/{name}_2styles.png)")

        # Cas 3 : sortie directe sous forme de tenseur [0, 1] pour un CNN en aval
        tensor_out = model.transfer(
            content=content_img,
            style=style_img_1,
            style2=style_img_2,
            size=256,
            return_tensor=True,
        )
        print(
            f"[{name}] Tenseur : shape={tuple(tensor_out.shape)}, "
            f"min={tensor_out.min().item():.3f}, max={tensor_out.max().item():.3f}"
        )


if __name__ == "__main__":
    main()
