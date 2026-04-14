from src.data_processing.tools.viz import resolve_wavelet_name


def print_transformation_params(params: dict) -> None:
    print()
    print("--------------------------------")
    print("Parametres transformation :")
    print("--------------------------------")
    print(f"Duration  : {params['duration']}")
    print(f"Offset    : {params['offset']}")
    print(f"Transform : {params['transform']}")
    print(f"Wavelet   : {resolve_wavelet_name(params)}")
    print("--------------------------------")
    print()
