# API Documentation - API Systems

## Authentication

### OAuth2 Flow
```
POST /auth/token
Content-Type: application/json

{
  "grant_type": "client_credentials",
  "client_id": "your_client_id",
  "client_secret": "your_client_secret"
}
```

### API Key Authentication
```
GET /api/resource
X-API-Key: your_api_key
```

## Endpoints

### Health Check
```
GET /health
Response: 200 OK
{
  "status": "healthy",
  "timestamp": "2025-06-06T21:40:00Z",
  "version": "1.0.0"
}
```

### System Status
```
GET /status
Response: 200 OK
{
  "system": "API Systems",
  "status": "operational",
  "uptime": 99.99,
  "last_check": "2025-06-06T21:40:00Z"
}
```

## Error Codes

- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Rate Limited
- 500: Internal Server Error

## Rate Limits

- Standard: 1000 requests/hour
- Premium: 10000 requests/hour
- Enterprise: Unlimited

Last Updated: 2025-06-06T22:00:24.291066
