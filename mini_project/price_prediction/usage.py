import joblib
import pandas as pd

from utils import get_app_root


def load_model(filename='used_car_price_model.joblib'):
    app_root = get_app_root()
    model_path = app_root / 'models' / filename
    return joblib.load(model_path)


def do_price_prediction(
    trained_model,
    brand,
    model,
    fuel_type,
    transmission_clean,
    brand_segment,
    ext_col_clean,
    int_col_clean,
    age,
    milage,
    horsepower,
    engine_liters,
    accident=0,  # Defaulting optional columns if applicable
    clean_title=1,
):
    raw_car_data = pd.DataFrame([{
        "brand": brand,
        "model": model,
        "fuel_type": fuel_type,
        "transmission_clean": transmission_clean,
        "brand_segment": brand_segment,
        "ext_col_clean": ext_col_clean,
        "int_col_clean": int_col_clean,
        "age": age,
        "milage": milage,
        "horsepower": horsepower,
        "engine_liters": engine_liters,
        "accident": accident,
        "clean_title": clean_title,
    }])

    predicted_array = trained_model.predict(raw_car_data)
    predicted_price = float(predicted_array[0])
    return round(predicted_price, 2)


def main():
    model_pipeline = load_model()
    estimated_valuation = do_price_prediction(
        trained_model=model_pipeline,
        brand="Ford",
        model="Mustang Base",
        fuel_type="Gasoline",
        transmission_clean="Automatic",
        brand_segment="Luxury",
        ext_col_clean="Black",
        int_col_clean="Black",
        age=3,
        milage=25000,
        horsepower=310.0,
        engine_liters=2.3,
        accident=0,
        clean_title=1,
    )

    print(f"💰 Estimated Market Value: ${estimated_valuation:,.2f}")


if __name__ == "__main__":
    main()