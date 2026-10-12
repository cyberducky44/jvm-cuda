public final class IntegerMath {
    public static int sum(int limit) {
        int total = 0;
        for (int i = 0; i < limit; i++) {
            total += i;
        }
        return total;
    }

    public static int mix(int x, int y) {
        return ((x + y) * 31 - (x - y)) ^ ((x & y) | (x ^ y));
    }

    public static int divide(int x, int y) { return x / y; }
    public static int remainder(int x, int y) { return x % y; }
    public static int negate(int x) { return -x; }
    public static int left(int x, int distance) { return x << distance; }
    public static int right(int x, int distance) { return x >> distance; }
    public static int unsignedRight(int x, int distance) { return x >>> distance; }

    public static int compare(int x, int y) {
        int result = 0;
        if (x == y) result += 1;
        if (x != y) result += 2;
        if (x < y) result += 4;
        if (x >= y) result += 8;
        if (x > y) result += 16;
        if (x <= y) result += 32;
        if (x == 0) result += 64;
        if (x != 0) result += 128;
        if (x < 0) result += 256;
        if (x >= 0) result += 512;
        if (x > 0) result += 1024;
        if (x <= 0) result += 2048;
        return result;
    }

    public static int denseSwitch(int x) {
        switch (x) {
            case -2: return 101;
            case -1: return -202;
            case 0: return 303;
            case 1: return -404;
            default: return 505;
        }
    }

    public static int sparseSwitch(int x) {
        switch (x) {
            case -2000000000: return -1;
            case 17: return 2;
            case 1000000000: return 3;
            default: return 4;
        }
    }

    public static int increment(int x) {
        x += 300;
        x -= 30000;
        x++;
        return x;
    }

    public static int constant() { return -2147483648; }
    public static int spin(int x) { while (true) { x++; } }
}
