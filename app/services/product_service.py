from app.database import df

def get_products():
    products = df.head(100)
    return products.drop_duplicates(subset='ProductID')

def get_product_by_id(productId: int):
    producto = df.loc[df['ProductID'] == productId].to_dict(orient='records')
    return producto[0] if producto else None