SELECT *
FROM orders
WHERE order_purchase_timestamp > (
    SELECT date(MAX(order_purchase_timestamp), '-3 months')
    FROM orders
)
AND order_status != "canceled"
AND order_delivered_customer_date > date(order_estimated_delivery_date, '+3 days');


SELECT seller_id, SUM((price + freight_value) * order_item_id) AS sales_revenues
FROM order_items
GROUP BY seller_id
HAVING sales_revenues > 100000;


SELECT seller_id, SUM(order_item_id) AS nb_product_sold
FROM order_items JOIN orders
ON (
    order_items.order_id = orders.order_id

    AND order_purchase_timestamp > (
        SELECT date(MAX(order_purchase_timestamp), '-3 months')
        FROM orders
    )
)
GROUP BY seller_id
HAVING nb_product_sold > 30;


SELECT customer_zip_code_prefix, AVG(review_score)
FROM order_reviews JOIN orders JOIN customers
ON (
    order_reviews.order_id = orders.order_id
    AND orders.customer_id = customers.customer_id
    AND review_creation_date > (
        SELECT date(MAX(review_creation_date), '-12 months')
        FROM order_reviews
    )
)
GROUP BY customer_zip_code_prefix
HAVING COUNT(review_id) >= 30
ORDER BY AVG(review_score)
LIMIT 5;





